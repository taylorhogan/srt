// The white backing for the north camera's roof AprilTag print
// (north_roof_tag_black_id2.scad),
// a plain filled square to lay the black cells on (2026-11-05).
//
// Same size as the whole tag: 150 mm square, which is the 8 x 8 black region
// (120 mm) plus one module (15 mm) of white quiet zone on every side. Center the black piece on it so that border is even all round; the
// detector needs at least one module of white on every side to find the
// tag's outline.
//
// Printing:
//   * White filament, matte face up (same reason as the black piece: a glossy
//     face throws the IR illuminator's highlight back at the lens).
//   * White is fine in the near infrared for nearly every filament, but check
//     the finished tag under the camera at night before you mount it.

tag_size  = 150;              // whole tag including the white quiet zone
thickness = 1.6;              // mm

cube([tag_size, tag_size, thickness]);
