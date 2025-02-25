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




// In the environment probe pixel shader:
struct ProbeBufferOutput 
{
    float4 Specular : SV_Target0;  // RGB: Specular radiance, A: Probe fade/weight
    float4 Diffuse : SV_Target1;   // RGB: Diffuse irradiance, A: Probe fade/weight 
};



META_PS(true, FEATURE_LEVEL_ES2)
ProbeBufferOutput PS_EnvProbe(Model_VS2PS input)
{
    ProbeBufferOutput output = (ProbeBufferOutput)0;
    
    // Sample GBuffer
    float2 uv = (input.ScreenPos.xy / input.ScreenPos.w) * float2(0.5, -0.5) + float2(0.5, 0.5);
    GBufferData gBufferData = GetGBufferData();
    GBufferSample gBuffer = SampleGBuffer(gBufferData, uv);
    
    if (gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
    {
        discard;
        return output;
    }
    
    float minpercent = 0.8f;
    float3 captureVector = gBuffer.WorldPos - PData.ProbePos;
    float distance = length(captureVector);
    float radius = 1.0 / PData.ProbeInvRadius;
    
    // Perfect blend weight calculation
    float normalizedDist = saturate(distance / radius);
    float t = saturate((normalizedDist - minpercent) / (1.0 - minpercent));
    float weight = (1.0 - t * t * (3.0 - 2.0 * t)) * PData.ProbeBrightness; // Smoothstep for C2 continuity
    
    if (weight <= 0.0001f)
    {
        discard;
        return output;
    }

    // Sample for specular reflections
    float3 V = normalize(gBuffer.WorldPos - gBufferData.ViewPos);
    float3 R = reflect(V, gBuffer.Normal);
    float3 specularDir = normalize(captureVector + R / PData.ProbeInvRadius);
    float3 specularSample = Probe.SampleLevel(SamplerLinearClamp, specularDir, ProbeMipFromRoughness(gBuffer.Roughness)).rgb;
    output.Specular = float4(specularSample * weight, weight);
    
    // Sample for diffuse reflections
    float3 probeSpaceNormal = normalize(captureVector + gBuffer.Normal / PData.ProbeInvRadius);
    float3 diffuseSample = Probe.SampleLevel(SamplerLinearClamp, probeSpaceNormal, REFLECTION_CAPTURE_NUM_MIPS - 1).rgb;
    output.Diffuse = float4(diffuseSample * weight, weight);
    
    return output;
}



META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_CombinePass(Quad_VS2PS input) : SV_Target0
{
    GBufferData gBufferData = GetGBufferData();
    GBufferSample gBuffer = SampleGBuffer(gBufferData, input.TexCoord);
    
    if (gBuffer.ShadingModel == SHADING_MODEL_UNLIT)
        return 0;

    float4 specularProbe = SAMPLE_RT(Reflections, input.TexCoord);
    float4 diffuseProbe = SAMPLE_RT(DiffuseReflections, input.TexCoord);

    // Use proper weighted average blending - prevent division by zero
    float specularWeight = max(specularProbe.a, 0.0001);
    float diffuseWeight = max(diffuseProbe.a, 0.0001);
    
    // Proper weighted average
    float3 specularResult = specularProbe.rgb / specularWeight;
    float3 diffuseResult = diffuseProbe.rgb / diffuseWeight;
    
    // Prepare lighting calculations
    float3 V = normalize(gBufferData.ViewPos - gBuffer.WorldPos);
    float NoV = saturate(dot(gBuffer.Normal, V));

    float3 specularColor = GetSpecularColor(gBuffer);
    float3 diffuseColor = GetDiffuseColor(gBuffer);
    
    float roughnessSq = gBuffer.Roughness * gBuffer.Roughness;
    float specularOcclusion = GetSpecularOcclusion(NoV, roughnessSq, gBuffer.AO);

    float3 specularResponse = EnvBRDF(PreIntegratedGF, specularColor, gBuffer.Roughness, NoV) * specularOcclusion;
    float3 diffuseResponse = diffuseColor * gBuffer.AO;

    return float4(
        specularResult * specularResponse + 
        diffuseResult * diffuseResponse, 
        0);
}
