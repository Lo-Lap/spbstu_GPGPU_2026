cbuffer SceneCB : register(b0)
{
    row_major float4x4 gMvp;
    float2 gInvViewport;
    float gPointSize;
    float gPadding;
};

struct VSInput
{
    float3 pos : POSITION;
    float4 color : COLOR0;
};

struct PSInput
{
    float4 pos : SV_POSITION;
    float4 color : COLOR0;
};

PSInput main(VSInput input)
{
    PSInput output;
    output.pos = mul(float4(input.pos, 1.0f), gMvp);
    output.color = input.color;
    return output;
}
