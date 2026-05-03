#include "compute/utils.cuh"

namespace compute {

int ceil_div(int n, int d) { return (n + d - 1) / d; }

Region aligned_brush_dab_region(BrushDab brush_dab, Size grid_size) {
  // Only grid vertices strictly inside the brush dab circle are selected
  int col_start = std::max(0, static_cast<int>(std::floor(brush_dab.circle.center.x - brush_dab.circle.radius + 1)));
  int col_end = std::min(static_cast<int>(std::ceil(brush_dab.circle.center.x + brush_dab.circle.radius - 1)),
                         grid_size.width - 1);
  int row_start = std::max(0, static_cast<int>(std::floor(brush_dab.circle.center.y - brush_dab.circle.radius + 1)));
  int row_end = std::min(static_cast<int>(std::ceil(brush_dab.circle.center.y + brush_dab.circle.radius - 1)),
                         grid_size.height - 1);

  // Align col start and end to 16 bytes for float4 access
  int start_remainder = col_start % 4;
  col_start -= start_remainder;
  int end_remainder = col_end % 4;
  col_end += 3 - end_remainder;

  Vertex origin{.col = col_start, .row = row_start};
  Size size{.width = col_end - col_start + 1, .height = row_end - row_start + 1};

  return Region{
      .origin = origin,
      .size = size,
  };
}

} // namespace compute