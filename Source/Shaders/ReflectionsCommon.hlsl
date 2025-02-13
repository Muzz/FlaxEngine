// Copyright (c) 2012-2024 Wojciech Figat. All rights reserved.

#ifndef __REFLECTIONS_COMMON__
#define __REFLECTIONS_COMMON__

#include "./Flax/GBufferCommon.hlsl"

// Hit depth (view space) threshold to detect if sky was hit (value above it where 1.0f is default)
#define REFLECTIONS_HIT_THRESHOLD 0.9f

float GetSpecularOcclusion(float NoV, float roughnessSq, float ao)
{
    return saturate(pow(NoV + ao, roughnessSq) - 1 + ao);
}

float4 SampleReflectionProbe(float3 viewPos, TextureCube probe, ProbeData data, float3 positionWS, float3 normal, float roughness)
{
    // Distance fade
    float3 captureVector = positionWS - data.ProbePos;
    float captureVectorLength = length(captureVector);
    float normalizedDistanceToCapture = saturate(captureVectorLength * data.ProbeInvRadius);
    float fade = (1.0 - smoothstep(0.7, 1, normalizedDistanceToCapture)) * data.ProbeBrightness;
    
    if (fade <= 0.0f)
        return 0;

    // For diffuse (roughness = 1), we sample in normal direction
    // For specular, we sample in reflection direction
    // This smoothly transitions between the two based on roughness
    float3 V = normalize(positionWS - viewPos);
    float3 R = reflect(V, normal);
    float3 sampleDir = lerp(R, normal, roughness * roughness);
    sampleDir = data.ProbeInvRadius * captureVector + sampleDir;
    
    // Sample with appropriate mip level
    half mip = ProbeMipFromRoughness(roughness);
    float3 radiance = probe.SampleLevel(SamplerLinearClamp, sampleDir, mip).rgb;
    
    return float4(radiance * fade, fade);
}

struct ProbeIBLResult
{
    float3 Specular;
    float3 Diffuse;
    float Fade;
};

ProbeIBLResult SampleProbeIBL(float3 viewPos, TextureCube probe, ProbeData data, float3 positionWS, float3 normal, float roughness)
{
    ProbeIBLResult result = (ProbeIBLResult) 0;
    
    // Calculate distance from probe to the pixel
    float3 captureVector = positionWS - data.ProbePos;
    float captureVectorLength = length(captureVector);
    
    // Distance fade
    float normalizedDistanceToCapture = saturate(captureVectorLength * data.ProbeInvRadius);
    result.Fade = 1.0 - smoothstep(0.7, 1, normalizedDistanceToCapture);
    float fade = result.Fade * data.ProbeBrightness;

    // Early out if too far
    if (fade <= 0.0f)
        return result;

    // Calculate view and reflection vectors
    float3 V = normalize(positionWS - viewPos);
    float3 R = reflect(V, normal);
    float3 D = data.ProbeInvRadius * captureVector + R;
    
    // Sample specular reflection
    half specularMip = ProbeMipFromRoughness(roughness);
    result.Specular = probe.SampleLevel(SamplerLinearClamp, D, specularMip).rgb * fade;
    
    // Sample diffuse irradiance (at highest mip which should be pre-convolved for diffuse)
    float3 probeSpaceNormal = data.ProbeInvRadius * captureVector + normal;
    result.Diffuse = probe.SampleLevel(SamplerLinearClamp, probeSpaceNormal, REFLECTION_CAPTURE_NUM_MIPS - 1).rgb * fade;
    
    return result;
}

// Calculates the reflective environment lighting to multiply the raw reflection color for the specular light (eg. from Env Probe or SSR).
float3 GetReflectionSpecularLighting(float3 viewPos, GBufferSample gBuffer)
{
    float3 specularColor = GetSpecularColor(gBuffer);
    float3 V = normalize(viewPos - gBuffer.WorldPos);
    float NoV = saturate(dot(gBuffer.Normal, V));
    return EnvBRDFApprox(specularColor, gBuffer.Roughness, NoV);
}

#endif
