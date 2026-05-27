cbuffer SceneCB : register(b0)
{
    row_major float4x4 gMvp;
    float2 gInvViewport;
    float gPointSize;
    float gPadding;
};

struct VSOutput
{
    float3 pos : POSITION;
    float4 color : COLOR0;
};

struct PSInput
{
    float4 pos : SV_POSITION;
    float4 color : COLOR0;
};

[maxvertexcount(4)]
void main(point VSOutput input[1], inout TriangleStream<PSInput> stream)
{
    float4 center = mul(float4(input[0].pos, 1.0f), gMvp);
    if (center.w <= 0.001f)
    {
        return;
    }

    float2 offset = gPointSize * gInvViewport * center.w;

    PSInput v;
    v.color = input[0].color;

    v.pos = center + float4(-offset.x,  offset.y, 0.0f, 0.0f);
    stream.Append(v);
    v.pos = center + float4( offset.x,  offset.y, 0.0f, 0.0f);
    stream.Append(v);
    v.pos = center + float4(-offset.x, -offset.y, 0.0f, 0.0f);
    stream.Append(v);
    v.pos = center + float4( offset.x, -offset.y, 0.0f, 0.0f);
    stream.Append(v);
}
