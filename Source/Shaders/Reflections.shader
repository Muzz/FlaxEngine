// Copyright (c) 2012-2024 Wojciech Figat. All rights reserved.

#include "./Flax/Common.hlsl"
#include "./Flax/MaterialCommon.hlsl"
#include "./Flax/BRDF.hlsl"
#include "./Flax/Random.hlsl"
#include "./Flax/MonteCarlo.hlsl"
#include "./Flax/LightingCommon.hlsl"
#include "./Flax/GBuffer.hlsl"
#include "./Flax/ReflectionsCommon.hlsl"
#include "./Flax/BRDF.hlsl"

META_CB_BEGIN(0, Data)

ProbeData PData;
float4x4 WVP;
GBufferData GBuffer;

META_CB_END

DECLARE_GBUFFERDATA_ACCESS(GBuffer)

TextureCube Probe : register(t4);
Texture2D Reflections : register(t5);
Texture2D PreIntegratedGF : register(t6);
Texture2D DiffuseReflections : register(t7);  

// Vertex Shader for models rendering
META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, ALIGN, PER_VERTEX, 0, true)
Model_VS2PS VS_Model(ModelInput_PosOnly input)
{
	Model_VS2PS output;
	output.Position = mul(float4(input.Position.xyz, 1), WVP);
	output.ScreenPos = output.Position;
	return output;
}

/*
// Pixel Shader for enviroment probes rendering
META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_EnvProbe(Model_VS2PS input) : SV_Target0
{
	// Obtain UVs corresponding to the current pixel
	float2 uv = (input.ScreenPos.xy / input.ScreenPos.w) * float2(0.5, -0.5) + float2(0.5, 0.5);

	// Sample GBuffer
	GBufferData gBufferData = GetGBufferData();
	GBufferSample gBuffer = SampleGBuffer(gBufferData, uv);

	// Check if cannot light a pixel
	BRANCH
	if (gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
	{
		discard;
		return 0;
	}

	// Sample probe
	return SampleReflectionProbe(gBufferData.ViewPos, Probe, PData, gBuffer.WorldPos, gBuffer.Normal, gBuffer.Roughness);
}
*/

// In the environment probe pixel shader:
struct ProbeBufferOutput 
{
    float4 Specular : SV_Target0;  // RGB: Specular radiance, A: Probe fade/weight
    float4 Diffuse : SV_Target1;   // RGB: Diffuse irradiance, A: Probe fade/weight 
};


META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_EnvProbeOLD(Model_VS2PS input) : SV_Target0
{
    // Obtain UVs corresponding to the current pixel
    float2 uv = (input.ScreenPos.xy / input.ScreenPos.w) * float2(0.5, -0.5) + float2(0.5, 0.5);
    // Sample GBuffer
    GBufferData gBufferData = GetGBufferData();
    GBufferSample gBuffer = SampleGBuffer(gBufferData, uv);
    // Check if cannot light a pixel
    BRANCH
    if (gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
    {
        discard;
        return 0;
    }
    // Sample probe
    return SampleReflectionProbe(gBufferData.ViewPos, Probe, PData, gBuffer.WorldPos, gBuffer.Normal, gBuffer.Roughness);
}

META_PS(true, FEATURE_LEVEL_ES2)
ProbeBufferOutput PS_EnvProbe(Model_VS2PS input)
{
    ProbeBufferOutput output = (ProbeBufferOutput)0;
    
    // Sample GBuffer
    float2 uv = (input.ScreenPos.xy / input.ScreenPos.w) * float2(0.5, -0.5) + float2(0.5, 0.5);

    GBufferData gBufferData = GetGBufferData();
    GBufferSample gBuffer = SampleGBuffer(gBufferData, uv);
    
    // Probe sampling setup
    float3 captureVector = gBuffer.WorldPos - PData.ProbePos;
    float captureVectorLength = length(captureVector);
    float fade = 1.0 - smoothstep(0.7, 1, saturate(captureVectorLength * PData.ProbeInvRadius));
    fade *= PData.ProbeBrightness;

    if (fade <= 0.0f || gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
    {
        discard;
        return output;
    }


    float3 V = normalize(gBuffer.WorldPos - gBufferData.ViewPos); // Flipped to point TO surface FROM camera
    float3 R = reflect(V, gBuffer.Normal);
    float3 specularDir = normalize(captureVector + R / PData.ProbeInvRadius);
    output.Specular = float4(Probe.SampleLevel(SamplerLinearClamp, specularDir, ProbeMipFromRoughness(gBuffer.Roughness)).rgb * fade, fade);

    // Diffuse sampling (going to specular output as in original working version)
    float3 probeSpaceNormal = normalize(captureVector + gBuffer.Normal / PData.ProbeInvRadius);
    output.Diffuse = float4(Probe.SampleLevel(SamplerLinearClamp, probeSpaceNormal, REFLECTION_CAPTURE_NUM_MIPS - 1).rgb * fade, fade);

    //output.Specular = float4(0.0,0.0,0.0,0.0);
    //output.Diffuse = float4(0.0,0.0,0.0,0.0);
    return output;
}

/*
// Pixel Shader for reflections combine pass (additive rendering to the light buffer)
META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_CombinePass(Quad_VS2PS input) : SV_Target0
{
	// Sample GBuffer
	GBufferData gBufferData = GetGBufferData();
	GBufferSample gBuffer = SampleGBuffer(gBufferData, input.TexCoord);

	// Check if cannot light pixel
	BRANCH
	if (gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
	{
		return 0;
	}

	// Sample reflections buffer
	float3 reflections = SAMPLE_RT(Reflections, input.TexCoord).rgb;

	// Calculate specular color
	float3 specularColor = GetSpecularColor(gBuffer);
	if (gBuffer.Metalness < 0.001)
		specularColor = 0.04f * gBuffer.Specular;

	// Calculate reflecion color
	float3 V = normalize(gBufferData.ViewPos - gBuffer.WorldPos);
	float NoV = saturate(dot(gBuffer.Normal, V));
	reflections *= EnvBRDF(PreIntegratedGF, specularColor, gBuffer.Roughness, NoV);

	// Apply specular occlusion
	float roughnessSq = gBuffer.Roughness * gBuffer.Roughness;
	float specularOcclusion = GetSpecularOcclusion(NoV, roughnessSq, gBuffer.AO);
	reflections *= specularOcclusion;

	return float4(reflections, 0);
}
*/

// In the combine pass:
META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_CombinePass(Quad_VS2PS input) : SV_Target0
{
    GBufferData gBufferData = GetGBufferData();
    GBufferSample gBuffer = SampleGBuffer(gBufferData, input.TexCoord);
    
    if (gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
        return 0;

    float4 specularProbe = SAMPLE_RT(Reflections, input.TexCoord);
    float4 diffuseProbe = SAMPLE_RT(DiffuseReflections, input.TexCoord); // New texture binding needed

    float3 V = normalize(gBufferData.ViewPos - gBuffer.WorldPos);
    float NoV = saturate(dot(gBuffer.Normal, V));

    float3 specularColor = GetSpecularColor(gBuffer);
    float3 diffuseColor = GetDiffuseColor(gBuffer);
    
    float roughnessSq = gBuffer.Roughness * gBuffer.Roughness;
    float specularOcclusion = GetSpecularOcclusion(NoV, roughnessSq, gBuffer.AO);

    float3 specularResponse = EnvBRDF(PreIntegratedGF, specularColor, gBuffer.Roughness, NoV) * specularOcclusion;
    float3 diffuseResponse = diffuseColor * gBuffer.AO;

    return float4(
        specularProbe.rgb * specularResponse + 
        diffuseProbe.rgb * diffuseResponse, 
        0);
}
