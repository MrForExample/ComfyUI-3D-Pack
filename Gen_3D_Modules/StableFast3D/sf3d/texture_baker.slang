// xy: 2D test position
// v1: vertex position 1
// v2: vertex position 2
// v3: vertex position 3
//
bool barycentric_coordinates(float2 xy, float2 v1, float2 v2, float2 v3, out float u, out float v, out float w)
{
    // Return true if the point (x,y) is inside the triangle defined by the vertices v1, v2, v3.
    // If the point is inside the triangle, the barycentric coordinates are stored in u, v, and w.
    float2 v1v2 = v2 - v1;
    float2 v1v3 = v3 - v1;
    float2 xyv1 = xy - v1;

    float d00 = dot(v1v2, v1v2);
    float d01 = dot(v1v2, v1v3);
    float d11 = dot(v1v3, v1v3);
    float d20 = dot(xyv1, v1v2);
    float d21 = dot(xyv1, v1v3);

    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;

    return (v >= 0.0) && (w >= 0.0) && (v + w <= 1.0);
}

[AutoPyBindCUDA]
[CUDAKernel]
void interpolate(
    TensorView<float3> attr,
    TensorView<int3> indices,
    TensorView<float4> rast,
    TensorView<float3> output)
{
    // Interpolate the attr into output based on the rast result (barycentric coordinates, + triangle idx)

    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (dispatch_id.x > output.size(0) || dispatch_id.y > output.size(1))
        return;

    float4 barycentric = rast[dispatch_id.x, dispatch_id.y];
    int triangle_idx = int(barycentric.w);

    if (triangle_idx < 0) {
        output[dispatch_id.x, dispatch_id.y] = float3(0.0, 0.0, 0.0);
        return;
    }

    float3 v1 = attr[indices[triangle_idx].x];
    float3 v2 = attr[indices[triangle_idx].y];
    float3 v3 = attr[indices[triangle_idx].z];

    output[dispatch_id.x, dispatch_id.y] = v1 * barycentric.x + v2 * barycentric.y + v3 * barycentric.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void bake_uv(
    TensorView<float2> uv,
    TensorView<int3> indices,
    TensorView<float4> output)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (dispatch_id.y > output.size(0) || dispatch_id.x > output.size(1))
        return;

    // We index x,y but the orginal coords are HW. So swap them
    float2 pixel_coord = float2(dispatch_id.y, dispatch_id.x);
    // Normalize to [0, 1]
    pixel_coord /= float2(output.size(1), output.size(0));
    pixel_coord = clamp(pixel_coord, 0.0, 1.0);
    // Flip x-axis
    pixel_coord.y = 1 - pixel_coord.y;

    for (int i = 0; i < indices.size(0); i++) {
        float2 v1 = float2(uv[indices[i].x].x, uv[indices[i].x].y);
        float2 v2 = float2(uv[indices[i].y].x, uv[indices[i].y].y);
        float2 v3 = float2(uv[indices[i].z].x, uv[indices[i].z].y);

        float u, v, w;
        bool hit = barycentric_coordinates(pixel_coord, v1, v2, v3, u, v, w);

        if (hit){
            output[dispatch_id.x, dispatch_id.y] = float4(u, v, w, i);
            return;
        }
    }

    output[dispatch_id.x, dispatch_id.y] = float4(0.0, 0.0, 0.0, -1);
}
