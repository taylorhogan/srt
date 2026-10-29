// The scope's AprilTag (36h11, id 0) as a 3D print: the BLACK cells only,
// one connected piece, to lay on a white background (2026-10-29).
//
// Same tag as the paper one on the scope: 4" (101.6 mm) including its one
// module of white quiet zone, so 10 modules of 10.16 mm, and this part -- the
// 8 x 8 black region -- is 81.28 mm square. The white quiet zone is the
// background you lay it on: leave at least one module (10.16 mm) of white
// showing on every side, or the detector cannot find the tag's outline.
//
// One piece: the border ring and every black data cell already touch along
// an edge, except the cell at row 3, column 5 (row 0 = top), which meets its
// neighbors only at corners. A small square patch sits on each of its three
// shared corners. The detector ignores the outer 13% of every cell when it
// reads it (OpenCV's perspectiveRemoveIgnoredMarginPerCell = 0.13, 1.32 mm
// here), and a 1.6 mm patch reaches only 0.8 mm into the white cells beside
// it, so the decode does not see it. Verified by rendering this geometry and
// decoding it as id 0 (scripts/scope_tag_scad_check.py).
//
// Printing:
//   * Black filament that is black in the NEAR INFRARED. The inside camera
//     reads this tag under its IR illuminator most of the night, and some
//     black dyes are close to transparent there -- the tag looks perfect by
//     day and vanishes after dark. Check a test square under the camera at
//     night before you mount it.
//   * Matte face: print the face that will show DOWN on a textured sheet, or
//     on the bare bed and scuff it lightly. A glossy top throws the
//     illuminator's highlight straight back at the lens.
//   * The pattern is not mirror-symmetric. If you print it face down, the
//     printed part is already the right way round when you flip it over --
//     mirror_for_face_down below does that for you.

tag_size  = 101.6;            // whole tag including the white quiet zone (4")
modules   = 10;               // 8 tag modules + 1 quiet module each side
module_mm = tag_size / modules;   // 10.16
thickness = 1.6;              // mm
patch     = 1.6;              // corner patch, mm (keep under 2 x 1.32)
mirror_for_face_down = true;  // true when the visible face prints on the bed
eps = 0.01;                   // overlap so touching cells fuse into one solid

// 1 = black, row 0 at the TOP as the tag is seen; tag 36h11 id 0.
bits = [
  [1,1,1,1,1,1,1,1],
  [1,1,1,0,1,1,1,1],
  [1,1,0,0,1,0,1,1],
  [1,1,1,1,0,1,0,1],
  [1,1,1,1,0,0,1,1],
  [1,0,1,0,0,0,1,1],
  [1,0,1,0,1,0,0,1],
  [1,1,1,1,1,1,1,1],
];

// Lattice corners [row, col] (row 0 = top edge) that get a patch.
corner_patches = [[3,5], [3,6], [4,6]];

N = len(bits);

module cell(r, c) {
  // x right, y up; row 0 at the top
  translate([c * module_mm - eps, (N - 1 - r) * module_mm - eps, 0])
    cube([module_mm + 2 * eps, module_mm + 2 * eps, thickness]);
}

module corner_patch(r, c) {
  translate([c * module_mm - patch / 2, (N - r) * module_mm - patch / 2, 0])
    cube([patch, patch, thickness]);
}

module tag_black() {
  union() {
    for (r = [0 : N - 1], c = [0 : N - 1])
      if (bits[r][c] == 1) cell(r, c);
    for (p = corner_patches) corner_patch(p[0], p[1]);
  }
}

if (mirror_for_face_down)
  translate([N * module_mm, 0, 0]) mirror([1, 0, 0]) tag_black();
else
  tag_black();
