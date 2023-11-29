// Copyright (c) 2012-2023 Wojciech Figat. All rights reserved.

#include "Font.h"
#include "FontAsset.h"
#include "FontManager.h"
#include "Engine/Core/Log.h"
#include "Engine/Threading/Threading.h"
#include "IncludeFreeType.h"
#include "MultiFont.h"

Font::Font(FontAsset* parentAsset, float size)
    : ManagedScriptingObject(SpawnParams(Guid::New(), Font::TypeInitializer))
    , _asset(parentAsset)
    , _size(size)
    , _characters(512)
{
    _asset->_fonts.Add(this);

    // Cache data
    FlushFaceSize();
    const FT_Face face = parentAsset->GetFTFace();
    ASSERT(face != nullptr);
    _height = Convert26Dot6ToRoundedPixel<int32>(FT_MulFix(face->height, face->size->metrics.y_scale));
    _hasKerning = FT_HAS_KERNING(face) != 0;
    _ascender = Convert26Dot6ToRoundedPixel<int16>(face->size->metrics.ascender);
    _descender = Convert26Dot6ToRoundedPixel<int16>(face->size->metrics.descender);
    _lineGap = _height - _ascender + _descender;
}

Font::~Font()
{
    if (_asset)
        _asset->_fonts.Remove(this);
}

void Font::GetCharacter(Char c, FontCharacterEntry& result)
{
    // Try to get the character or cache it if cannot be found
    if (!_characters.TryGet(c, result))
    {
        // This thread race condition may happen in editor but in game we usually do all stuff with fonts on main thread (chars caching)
        ScopeLock lock(_asset->Locker);

        // Handle situation when more than one thread wants to get the same character
        if (_characters.TryGet(c, result))
            return;

        // Create character cache
        FontManager::AddNewEntry(this, c, result);

        // Add to the dictionary
        _characters.Add(c, result);
    }
}

int32 Font::GetKerning(Char first, Char second) const
{
    int32 kerning = 0;
    const uint32 key = (uint32)first << 16 | second;
    if (_hasKerning && !_kerningTable.TryGet(key, kerning))
    {
        // This thread race condition may happen in editor but in game we usually do all stuff with fonts on main thread (chars caching)
        ScopeLock lock(_asset->Locker);

        // Handle situation when more than one thread wants to get the same character
        if (!_kerningTable.TryGet(key, kerning))
        {
            const FT_Face face = _asset->GetFTFace();
            ASSERT(face);

            FlushFaceSize();

            FT_Vector vec;
            const FT_UInt firstIndex = FT_Get_Char_Index(face, first);
            const FT_UInt secondIndex = FT_Get_Char_Index(face, second);
            FT_Get_Kerning(face, firstIndex, secondIndex, FT_KERNING_DEFAULT, &vec);
            kerning = vec.x >> 6;

            _kerningTable.Add(key, kerning);
        }
    }

    return kerning;
}

void Font::CacheText(const StringView& text)
{
    FontCharacterEntry entry;
    for (int32 i = 0; i < text.Length(); i++)
    {
        GetCharacter(text[i], entry);
    }
}

void Font::Invalidate()
{
    ScopeLock lock(_asset->Locker);

    for (auto i = _characters.Begin(); i.IsNotEnd(); ++i)
    {
        FontManager::Invalidate(i->Value);
    }
    _characters.Clear();
}

void Font::ProcessText(const StringView& text, Array<FontLineCache>& outputLines, const TextLayoutOptions& layout)
{
    float cursorX = 0;
    int32 kerning;
    FontLineCache tmpLine;
    FontCharacterEntry entry;
    FontCharacterEntry previous;
    int32 textLength = text.Length();
    float scale = layout.Scale / FontManager::FontScale;
    float boundsWidth = layout.Bounds.GetWidth();
    float baseLinesDistance = static_cast<float>(_height) * layout.BaseLinesGapScale * scale;
    tmpLine.Location = Float2::Zero;
    tmpLine.Size = Float2::Zero;
    tmpLine.FirstCharIndex = 0;
    tmpLine.LastCharIndex = -1;

    if (textLength == 0) {
        return;
    }

    int32 lastWrapCharIndex = INVALID_INDEX;
    float lastWrapCharX = 0;
    bool lastMoveLine = false;

    // Process each character to split text into single lines
    for (int32 currentIndex = 0; currentIndex < textLength;)
    {
        bool moveLine = false;
        float xAdvance = 0;
        int32 nextCharIndex = currentIndex + 1;

        // Submit line if text ends
        if (nextCharIndex == textLength) {
            moveLine = true;
        }

        // Cache current character
        const Char currentChar = text[currentIndex];
        const bool isWhitespace = StringUtils::IsWhitespace(currentChar);

        // Check if character can wrap words
        const bool isWrapChar = !StringUtils::IsAlnum(currentChar) || isWhitespace || StringUtils::IsUpper(currentChar);
        if (isWrapChar && currentIndex != 0)
        {
            lastWrapCharIndex = currentIndex;
            lastWrapCharX = cursorX;
        }

        // Check if it's a newline character
        if (currentChar == '\n')
        {
            // Break line
            moveLine = true;
            tmpLine.LastCharIndex++;
        }
        else
        {
            // Get character entry
            GetCharacter(currentChar, entry);

            // Get kerning
            if (!isWhitespace && previous.IsValid)
            {
                kerning = GetKerning(previous.Character, entry.Character);
            }
            else
            {
                kerning = 0;
            }
            previous = entry;
            xAdvance = (kerning + entry.AdvanceX) * scale;

            // Check if character fits the line or skip wrapping
            if (cursorX + xAdvance <= boundsWidth || layout.TextWrapping == TextWrapping::NoWrap)
            {
                // Move character
                cursorX += xAdvance;
                tmpLine.LastCharIndex++;
            }
            else if (layout.TextWrapping == TextWrapping::WrapWords)
            {
                if (lastWrapCharIndex != INVALID_INDEX)
                {
                    // Skip moving twice for the same character
                    int32 lastLineLastCharIndex = outputLines.HasItems() ? outputLines.Last().LastCharIndex : -10000;
                    if (lastLineLastCharIndex == lastWrapCharIndex || lastLineLastCharIndex == lastWrapCharIndex - 1 || lastLineLastCharIndex == lastWrapCharIndex - 2)
                    {
                        currentIndex = nextCharIndex;
                        lastMoveLine = moveLine;
                        continue;
                    }

                    // Move line
                    const Char wrapChar = text[lastWrapCharIndex];
                    moveLine = true;
                    cursorX = lastWrapCharX;
                    if (StringUtils::IsWhitespace(wrapChar))
                    {
                        // Skip whitespaces
                        tmpLine.LastCharIndex = lastWrapCharIndex - 1;
                        nextCharIndex = currentIndex = lastWrapCharIndex + 1;
                    }
                    else
                    {
                        tmpLine.LastCharIndex = lastWrapCharIndex - 1;
                        nextCharIndex = currentIndex = lastWrapCharIndex;
                    }
                }
            }
            else if (layout.TextWrapping == TextWrapping::WrapChars)
            {
                // Move line
                moveLine = true;
                nextCharIndex = currentIndex;

                // Skip moving twice for the same character
                if (lastMoveLine)
                    break;
            }
        }

        // Check if move to another line
        if (moveLine)
        {
            // Add line
            tmpLine.Size.X = cursorX;
            tmpLine.Size.Y = baseLinesDistance;
            tmpLine.LastCharIndex = Math::Max(tmpLine.LastCharIndex, tmpLine.FirstCharIndex);
            outputLines.Add(tmpLine);

            // Reset line
            tmpLine.Location.Y += baseLinesDistance;
            tmpLine.FirstCharIndex = nextCharIndex;
            tmpLine.LastCharIndex = nextCharIndex - 1;
            cursorX = 0;
            lastWrapCharIndex = INVALID_INDEX;
            lastWrapCharX = 0;
            previous.IsValid = false;
        }

        currentIndex = nextCharIndex;
        lastMoveLine = moveLine;
    }

    // Check if an additional line should be created
    if (text[textLength - 1] == '\n')
    {
        // Add line
        tmpLine.Size.X = cursorX;
        tmpLine.Size.Y = baseLinesDistance;
        tmpLine.LastCharIndex = textLength - 1;
        outputLines.Add(tmpLine);

        tmpLine.Location.Y += baseLinesDistance;
    }

    // Check amount of lines
    if (outputLines.IsEmpty())
        return;

    float totalHeight = tmpLine.Location.Y;

    Float2 offset = Float2::Zero;
    if (layout.VerticalAlignment == TextAlignment::Center)
    {
        offset.Y += (layout.Bounds.GetHeight() - totalHeight) * 0.5f;
    }
    else if (layout.VerticalAlignment == TextAlignment::Far)
    {
        offset.Y += layout.Bounds.GetHeight() - totalHeight;
    }
    for (int32 i = 0; i < outputLines.Count(); i++)
    {
        FontLineCache& line = outputLines[i];
        Float2 rootPos = line.Location + offset;

        // Fix upper left line corner to match desire text alignment
        if (layout.HorizontalAlignment == TextAlignment::Center)
        {
            rootPos.X += (layout.Bounds.GetWidth() - line.Size.X) * 0.5f;
        }
        else if (layout.HorizontalAlignment == TextAlignment::Far)
        {
            rootPos.X += layout.Bounds.GetWidth() - line.Size.X;
        }

        line.Location = rootPos;
    }
}

void Font::ProcessText(const Array<Font*>& fonts, const StringView& text, Array<MultiFontLineCache>& outputLines, API_PARAM(Ref) const TextLayoutOptions& layout)
{
    float cursorX = 0;
    int32 kerning;
    MultiFontLineCache tmpLine;
    MultiFontSegmentCache tmpSegment;
    FontCharacterEntry entry;
    FontCharacterEntry previous;
    int32 textLength = text.Length();
    float scale = layout.Scale / FontManager::FontScale;
    float boundsWidth = layout.Bounds.GetWidth();
    float baseLinesDistanceScale = layout.BaseLinesGapScale * scale;

    tmpSegment.Location = Float2::Zero;
    tmpSegment.Height = 0;
    tmpSegment.FirstCharIndex = 0;
    tmpSegment.LastCharIndex = -1;

    tmpLine.Location = Float2::Zero;
    tmpLine.Size = Float2::Zero;
    tmpLine.Segments = Array<MultiFontSegmentCache>();

    if (textLength == 0) {
        return;
    }

    int32 lastWrapCharIndex = INVALID_INDEX;
    float lastWrapCharX = 0;
    bool lastMoveLine = false;
    // The index of the font used by the current segment
    int32 currentFontIndex = GetCharFontIndex(fonts, text[0], 0);
    // The maximum font height of the current line
    float maxHeight = 0;
    float maxAscender = 0;

    // Process each character to split text into single lines
    for (int32 currentIndex = 0; currentIndex < textLength;)
    {
        bool moveLine = false;
        bool moveSegment = false;
        float xAdvance = 0;
        int32 nextCharIndex = currentIndex + 1;

        // Submit line and segment if text ends
        if (nextCharIndex == textLength) {
            moveLine = moveSegment = true;
        }

        // Cache current character
        const Char currentChar = text[currentIndex];
        const bool isWhitespace = StringUtils::IsWhitespace(currentChar);

        // Check if character can wrap words
        const bool isWrapChar = !StringUtils::IsAlnum(currentChar) || isWhitespace || StringUtils::IsUpper(currentChar) || (currentChar >= 0x3040 && currentChar <= 0x9FFF);
        if (isWrapChar && currentIndex != 0)
        {
            lastWrapCharIndex = currentIndex;
            lastWrapCharX = cursorX;
        }

        int32 nextFontIndex = currentFontIndex;
        // Check if it's a newline character
        if (currentChar == '\n')
        {
            // Break line
            moveLine = moveSegment = true;
            tmpSegment.LastCharIndex++;
        }
        else
        {
            // Get character entry
            if (nextCharIndex < textLength) {
                nextFontIndex = GetCharFontIndex(fonts, text[nextCharIndex], currentFontIndex);
            }

            // Get character entry
            fonts[currentFontIndex]->GetCharacter(currentChar, entry);
            maxHeight = Math::Max(maxHeight, static_cast<float>(fonts[currentFontIndex]->GetHeight()));
            maxAscender = Math::Max(maxAscender, static_cast<float>(fonts[currentFontIndex]->GetAscender()));

            // Move segment if the font changes or text ends
            if (nextFontIndex != currentFontIndex || nextCharIndex == textLength) {
                moveSegment = true;
            }

            // Get kerning, only when the font hasn't changed
            if (!isWhitespace && previous.IsValid && !moveSegment)
            {
                kerning = fonts[currentFontIndex]->GetKerning(previous.Character, entry.Character);
            }
            else
            {
                kerning = 0;
            }
            previous = entry;
            xAdvance = (kerning + entry.AdvanceX) * scale;

            // Check if character fits the line or skip wrapping
            if (cursorX + xAdvance <= boundsWidth || layout.TextWrapping == TextWrapping::NoWrap)
            {
                // Move character
                cursorX += xAdvance;
                tmpSegment.LastCharIndex++;
            }
            else if (layout.TextWrapping == TextWrapping::WrapWords)
            {
                if (lastWrapCharIndex != INVALID_INDEX)
                {
                    // Skip moving twice for the same character
                    int32 lastLineLastCharIndex = outputLines.HasItems() && outputLines.Last().Segments.HasItems() ? outputLines.Last().Segments.Last().LastCharIndex : -10000;
                    if (lastLineLastCharIndex == lastWrapCharIndex || lastLineLastCharIndex == lastWrapCharIndex - 1 || lastLineLastCharIndex == lastWrapCharIndex - 2)
                    {
                        currentIndex = nextCharIndex;
                        lastMoveLine = moveLine;
                        continue;
                    }

                    // Move line
                    const Char wrapChar = text[lastWrapCharIndex];
                    moveLine = true;
                    moveSegment = tmpSegment.FirstCharIndex < lastWrapCharIndex;

                    cursorX = lastWrapCharX;
                    if (StringUtils::IsWhitespace(wrapChar))
                    {
                        // Skip whitespaces
                        tmpSegment.LastCharIndex = lastWrapCharIndex - 1;
                        nextCharIndex = currentIndex = lastWrapCharIndex + 1;
                    }
                    else
                    {
                        tmpSegment.LastCharIndex = lastWrapCharIndex - 1;
                        nextCharIndex = currentIndex = lastWrapCharIndex;
                    }
                }
            }
            else if (layout.TextWrapping == TextWrapping::WrapChars)
            {
                // Move line
                moveLine = true;
                moveSegment = tmpSegment.FirstCharIndex < currentChar;
                nextCharIndex = currentIndex;

                // Skip moving twice for the same character
                if (lastMoveLine)
                    break;
            }
        }

        if (moveSegment) {
            // Add segment
            tmpSegment.Height = baseLinesDistanceScale * fonts[currentFontIndex]->GetHeight();
            tmpSegment.LastCharIndex = Math::Max(tmpSegment.LastCharIndex, tmpSegment.FirstCharIndex);
            tmpSegment.FontIndex = currentFontIndex;
            tmpLine.Segments.Add(tmpSegment);

            // Reset segment
            tmpSegment.Location.X = cursorX;
            tmpSegment.FirstCharIndex = nextCharIndex;
            tmpSegment.LastCharIndex = nextCharIndex - 1;

            currentFontIndex = nextFontIndex;
        }

        // Check if move to another line
        if (moveLine)
        {
            // Add line
            tmpLine.Size.X = cursorX;
            tmpLine.Size.Y = baseLinesDistanceScale * maxHeight;
            tmpLine.MaxAscender = maxAscender;
            outputLines.Add(tmpLine);

            // Reset line
            tmpLine.Segments.Clear();
            tmpLine.Location.Y += baseLinesDistanceScale * maxHeight;
            cursorX = 0;
            tmpSegment.Location.X = cursorX;
            lastWrapCharIndex = INVALID_INDEX;
            lastWrapCharX = 0;
            previous.IsValid = false;

            // Reset max font height
            maxHeight = 0;
            maxAscender = 0;
        }

        currentIndex = nextCharIndex;
        lastMoveLine = moveLine;
    }

    // Check if an additional line should be created
    if (text[textLength - 1] == '\n')
    {
        // Add line
        tmpLine.Size.X = cursorX;
        tmpLine.Size.Y = baseLinesDistanceScale * maxHeight;
        outputLines.Add(tmpLine);

        tmpLine.Location.Y += baseLinesDistanceScale * maxHeight;
    }

    // Check amount of lines
    if (outputLines.IsEmpty())
        return;

    float totalHeight = tmpLine.Location.Y;

    Float2 offset = Float2::Zero;
    if (layout.VerticalAlignment == TextAlignment::Center)
    {
        offset.Y += (layout.Bounds.GetHeight() - totalHeight) * 0.5f;
    }
    else if (layout.VerticalAlignment == TextAlignment::Far)
    {
        offset.Y += layout.Bounds.GetHeight() - totalHeight;
    }
    for (int32 i = 0; i < outputLines.Count(); i++)
    {
        MultiFontLineCache& line = outputLines[i];
        Float2 rootPos = line.Location + offset;

        // Fix upper left line corner to match desire text alignment
        if (layout.HorizontalAlignment == TextAlignment::Center)
        {
            rootPos.X += (layout.Bounds.GetWidth() - line.Size.X) * 0.5f;
        }
        else if (layout.HorizontalAlignment == TextAlignment::Far)
        {
            rootPos.X += layout.Bounds.GetWidth() - line.Size.X;
        }

        line.Location = rootPos;

        // Align all segments to center in case they have different heights
        for (int32 j = 0; j < line.Segments.Count(); j++)
        {
            MultiFontSegmentCache& segment = line.Segments[j];
            segment.Location.Y += (line.MaxAscender - fonts[segment.FontIndex]->GetAscender()) / 2;
        }
    }
}

Float2 Font::MeasureText(const StringView& text, const TextLayoutOptions& layout)
{
    // Check if there is no need to do anything
    if (text.IsEmpty())
        return Float2::Zero;

    // Process text
    Array<FontLineCache> lines;
    ProcessText(text, lines, layout);

    // Calculate bounds
    Float2 max = Float2::Zero;
    for (int32 i = 0; i < lines.Count(); i++)
    {
        const FontLineCache& line = lines[i];
        max = Float2::Max(max, line.Location + line.Size);
    }

    return max;
}

int32 Font::HitTestText(const StringView& text, const Float2& location, const TextLayoutOptions& layout)
{
    // Check if there is no need to do anything
    if (text.Length() <= 0)
        return 0;

    // Process text
    Array<FontLineCache> lines;
    ProcessText(text, lines, layout);
    ASSERT(lines.HasItems());
    float scale = layout.Scale / FontManager::FontScale;
    float baseLinesDistance = static_cast<float>(_height) * layout.BaseLinesGapScale * scale;

    // Offset position to match lines origin space
    Float2 rootOffset = layout.Bounds.Location + lines.First().Location;
    Float2 testPoint = location - rootOffset;

    // Get line which may intersect with the position (it's possible because lines have fixed height)
    int32 lineIndex = Math::Clamp(Math::FloorToInt(testPoint.Y / baseLinesDistance), 0, lines.Count() - 1);
    const FontLineCache& line = lines[lineIndex];
    float x = line.Location.X;

    // Check all characters in the line to find hit point
    FontCharacterEntry previous;
    FontCharacterEntry entry;
    int32 smallestIndex = INVALID_INDEX;
    float dst, smallestDst = MAX_float;
    for (int32 currentIndex = line.FirstCharIndex; currentIndex <= line.LastCharIndex; currentIndex++)
    {
        // Cache current character
        const Char currentChar = text[currentIndex];
        GetCharacter(currentChar, entry);
        const bool isWhitespace = StringUtils::IsWhitespace(currentChar);

        // Apply kerning
        if (!isWhitespace && previous.IsValid)
        {
            x += GetKerning(previous.Character, entry.Character);
        }
        previous = entry;

        // Test
        dst = Math::Abs(testPoint.X - x);
        if (dst < smallestDst)
        {
            // Found closer character
            smallestIndex = currentIndex;
            smallestDst = dst;
        }
        else if (dst > smallestDst)
        {
            // Current char is worse so return the best result
            return smallestIndex;
        }

        // Move
        x += entry.AdvanceX * scale;
    }

    // Test line end edge
    dst = Math::Abs(testPoint.X - x);
    if (dst < smallestDst)
    {
        // Pointer is behind the last character in the line
        smallestIndex = line.LastCharIndex;

        // Fix for last line
        if (lineIndex == lines.Count() - 1)
            smallestIndex++;
    }

    return smallestIndex;
}

Float2 Font::GetCharPosition(const StringView& text, int32 index, const TextLayoutOptions& layout)
{
    // Check if there is no need to do anything
    if (text.IsEmpty())
        return layout.Bounds.Location;

    // Process text
    Array<FontLineCache> lines;
    ProcessText(text, lines, layout);
    ASSERT(lines.HasItems());
    float scale = layout.Scale / FontManager::FontScale;
    float baseLinesDistance = static_cast<float>(_height) * layout.BaseLinesGapScale * scale;
    Float2 rootOffset = layout.Bounds.Location + lines.First().Location;

    // Find line with that position
    FontCharacterEntry previous;
    FontCharacterEntry entry;
    for (int32 lineIndex = 0; lineIndex < lines.Count(); lineIndex++)
    {
        const FontLineCache& line = lines[lineIndex];

        // Check if desire position is somewhere inside characters in line range
        if (Math::IsInRange(index, line.FirstCharIndex, line.LastCharIndex))
        {
            float x = line.Location.X;

            // Check all characters in the line
            for (int32 currentIndex = line.FirstCharIndex; currentIndex < index; currentIndex++)
            {
                // Cache current character
                const Char currentChar = text[currentIndex];
                GetCharacter(currentChar, entry);
                const bool isWhitespace = StringUtils::IsWhitespace(currentChar);

                // Apply kerning
                if (!isWhitespace && previous.IsValid)
                {
                    x += GetKerning(previous.Character, entry.Character);
                }
                previous = entry;

                // Move
                x += entry.AdvanceX * scale;
            }

            // Upper left corner of the character
            return rootOffset + Float2(x, static_cast<float>(lineIndex * baseLinesDistance));
        }
    }

    // Position after last character in the last line
    return rootOffset + Float2(lines.Last().Size.X, static_cast<float>((lines.Count() - 1) * baseLinesDistance));
}

bool Font::ContainsChar(Char c)
{
    return FT_Get_Char_Index(GetAsset()->GetFTFace(), c) > 0;
}

void Font::FlushFaceSize() const
{
    // Set the character size
    const FT_Face face = _asset->GetFTFace();
    const FT_Error error = FT_Set_Char_Size(face, 0, ConvertPixelTo26Dot6<FT_F26Dot6>(_size * FontManager::FontScale), DefaultDPI, DefaultDPI);
    if (error)
    {
        LOG_FT_ERROR(error);
    }

    // Clear transform
    FT_Set_Transform(face, nullptr, nullptr);
}

String Font::ToString() const
{
    return String::Format(TEXT("Font {0} {1}"), _asset ? _asset->GetFamilyName() : String::Empty, _size);
}
