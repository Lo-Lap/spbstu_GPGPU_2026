struct VSInput
{
    float3 pos : POSITION;
    float4 color : COLOR0;
};

struct VSOutput
{
    float3 pos : POSITION;
    float4 color : COLOR0;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    output.pos = input.pos;
    output.color = input.color;
    return output;
}
