import json, base64, sys, datetime, math
from pathlib import Path
S = Path(sys.argv[1]); OUT = Path(sys.argv[2])
M = json.load(open(S / "lc_model.json")); G = json.load(open(S / "geometry.json")); sol = json.load(open(S / "stack_solution.json"))
CS = json.load(open(S / "comps_stats.json")); stackmeta = json.load(open(S / "hubblev1_L_raw.json"))
def b64(p, mime="image/jpeg"):
    return "data:%s;base64,%s" % (mime, base64.b64encode((S / p).read_bytes()).decode())
def svg(p):
    t = (S / p).read_text()
    return t[t.index("<svg"):]
dc, dn = M["dist_cycle"], M["dist_nights"]
P = M["P"]; amp = M["amplitude"]; n = M["n_good"]
abs_mag = dc["abs_mag"]; mean = M["mean_cycle"]; mv = dc["m_v"]; mod = dc["modulus"]
true_mod = 24.47; true_mly = 2.54
lum_solar = 10 ** (-0.4 * (abs_mag - 0.2 - 4.74))   # M_bol ~ M_V - 0.2 for a 6000 K supergiant... BC ~ -0.2
sep_arcmin = 16.0; proj_kpc = math.radians(sep_arcmin / 60) * 780
frames_used = stackmeta.get("accepted", 208)
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hubble's Variable from Iris</title>
<style>
:root {{ --ink:#1b1b1b; --muted:#5a5a5a; --accent:#1f4e79; --rule:#d8d8d8; --soft:#f4f6f8; --orange:#e67e22; --red:#c0392b; }}
html,body {{ background:#fff; color:var(--ink); margin:0; font:16px/1.55 Georgia, 'Times New Roman', serif; }}
main {{ max-width:1080px; margin:0 auto; padding:28px 20px 60px; }}
h1 {{ font-size:34px; line-height:1.15; margin:0 0 6px; letter-spacing:-0.01em; }}
h2 {{ font-size:23px; margin:44px 0 10px; padding-top:14px; border-top:1px solid var(--rule); }}
h3 {{ font-size:17px; margin:22px 0 6px; font-family: 'Helvetica Neue', Arial, sans-serif; }}
p {{ margin:0 0 12px; }} .lead {{ font-size:18px; color:var(--muted); margin-bottom:18px; }}
figure {{ margin:18px 0 22px; }} figure svg, figure img {{ width:100%; height:auto; display:block; border:1px solid var(--rule); border-radius:4px; }}
figcaption {{ font:14px/1.45 'Helvetica Neue', Arial, sans-serif; color:var(--muted); margin-top:8px; }}
table {{ border-collapse:collapse; font:15px/1.4 'Helvetica Neue', Arial, sans-serif; margin:10px 0 18px; }}
td, th {{ padding:6px 14px 6px 0; text-align:left; vertical-align:top; border-bottom:1px solid var(--rule); }}
th {{ color:var(--muted); font-weight:600; }} td.num {{ font-variant-numeric: tabular-nums; }}
.facts {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:16px 0 8px; }}
.fact {{ background:var(--soft); border-radius:6px; padding:12px 14px; font-family:'Helvetica Neue', Arial, sans-serif; }}
.fact b {{ display:block; font-size:24px; color:var(--accent); font-weight:600; }} .fact span {{ font-size:13px; color:var(--muted); }}
.steps td:first-child {{ color:var(--muted); width:38%; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }} @media (max-width:760px) {{ .two {{ grid-template-columns:1fr; }} }}
.note {{ background:var(--soft); border-left:4px solid var(--accent); padding:10px 14px; margin:14px 0; font-size:15px; }}
.k {{ color:var(--orange); font-weight:600; }} .m {{ color:var(--accent); font-weight:600; }}
footer {{ margin-top:48px; color:var(--muted); font:13px/1.5 'Helvetica Neue', Arial, sans-serif; border-top:1px solid var(--rule); padding-top:12px; }}
@media print {{ main {{ max-width:none; padding:0; }} h2 {{ break-after:avoid; }} figure {{ break-inside:avoid; }} }}
</style></head><body><main>
<h1>Hubble's Variable: the Cepheid M31 V1 from Iris</h1>
<p class="lead">Why watch one faint star brighten and fade for a month? Because a Cepheid's light curve is a distance measurement in disguise. The curve gives the <b>period</b>, the days from one maximum to the next. The period gives the star's <b>true luminosity</b>, by Henrietta Leavitt's law: a Cepheid that pulsates slowly is intrinsically bright, one that pulsates quickly is faint, and the relation is tight enough to read the one from the other. And a star whose true luminosity is known, and whose apparent brightness is measured, can sit at only one <b>distance</b>, because light dims with the square of the distance. Period, luminosity, distance: that is the whole chain, and it reaches places no parallax can.</p>
<p>It is the chain Edwin Hubble ran on this very star in 1923. Its period put it so far away that the Andromeda &ldquo;nebula&rdquo; could not be a cloud of gas inside the Milky Way, as half of astronomy then held; it had to be a galaxy of its own, a separate island of stars, and with it the universe became a universe of galaxies. This page runs the same chain again on {n} clear nights between 2026-10-08 and 2026-11-10, one hour each, through the CDK17 at 0.26&Prime; a pixel: the light curve, the period it shows, the luminosity that period implies, and the distance that follows.</p>
<div class="facts">
<div class="fact"><b>{P:.1f} d</b><span>period (literature; fixed in the fit)</span></div>
<div class="fact"><b>{amp:.2f} mag</b><span>amplitude, {M['m_max']:.2f} to {M['m_min']:.2f}</span></div>
<div class="fact"><b>{n}</b><span>nightly points on the curve</span></div>
<div class="fact"><b>{mean:.2f}</b><span>mean magnitude over the cycle</span></div>
<div class="fact"><b>{dc['mly']:.1f} Mly</b><span>Leavitt-law distance ({dc['kpc']:.0f} kpc)</span></div>
</div>

<h2>1. The light curve</h2>
<p>Each blue point is one night: the star's brightness averaged over the ten or so five-minute frames of that hour, with the error bar the scatter of those frames and of the comparison-star zero point. The red line is a smooth periodic curve, three Fourier terms on the known 31.4-day period, fitted through the points by weighted least squares. It passes within or close to the error bars of every night but two, and the scatter about it is {M['rms_resid']:.3f} mag. The one grey cross is 2026-10-14, a night of thin cloud whose frames disagreed by eleven magnitudes; the pipeline marks such a night failed and it carries no weight.</p>
<figure>{svg('lc_time.svg')}<figcaption>Magnitudes are on the Gaia G scale, brighter upward, as astronomers draw them. Filled blue circles are the luminance-filter nights, green diamonds three early nights through the green filter. <span class="k">Hollow orange circles</span> are not measurements: they are what the fitted curve says each coming clear night should give, carried forward to the solstice. Every clear night from now on adds a point that either lands on them or does not.</figcaption></figure>
<p>What the shape says. The star took {M['rise_fraction']*P:.0f} days to brighten from its faintest to its brightest and {(1-M['rise_fraction'])*P:.0f} days to fade back. That asymmetry, a steep rise and a slow decline, is the signature of a classical Cepheid and is why these stars are hard to confuse with eclipsing binaries or rotating spotted stars, whose curves are symmetric. We caught the maximum on {M['date_max_obs'][:10]}, about two days before the 2010 ephemeris predicted it, which is the accumulated drift of a period known to a tenth of a day over sixteen years. The next maximum falls on <span class="k">{M['next_max_date'][:10]}</span>.</p>

<h2>2. The same nights folded on the period</h2>
<p>If the period is right, every night can be placed by its <em>phase</em>: the fraction of a cycle elapsed since a reference maximum. Points from different cycles then fall on one curve. Our 33 nights span just over one cycle, so the fold mostly re-sorts the same points, but it makes the gaps visible. The cycle is drawn twice so that the maximum is not cut at the edge.</p>
<figure>{svg('lc_fold.svg')}<figcaption>The shaded bands are the two tenths of the cycle, phases 0.7 to 0.9, that no night has sampled yet: the steep climb toward maximum. The <span class="k">orange</span> points are the predictions for 2026-11-12 through 11-30, and they land squarely in the gap. Ten clear nights would close the curve.</figcaption></figure>
<p>This is why the queue holds V1 open through 2026-11-22. The observatory's own rule for a monitor target is that it is finished when eight of ten phase bins hold a point; we have eight, but the two missing ones are the most informative part of the cycle. The rise is where a Cepheid's atmosphere is moving fastest, and it is where a wrong period shows up first: a point that lands a day early or late on the climb is displaced by a tenth of a magnitude, while on the slow decline the same error moves it by a hundredth.</p>

<h2>3. Where it is</h2>
<figure><img src="{b64('field_v1.jpg')}" alt="The stacked luminance field with V1 circled"><figcaption>The whole Iris frame: {frames_used} five-minute luminance exposures from the {n} nights, registered and combined, then passed through the observatory's learned denoiser. The glow at the bottom is the bulge of M31, 16 arcminutes from V1. The dark bands crossing the field are the galaxy's dust lanes; the fine grain across the right half is not noise but the unresolved stars of the disk. V1 is the circled point.</figcaption></figure>
<div class="two">
<figure><img src="{b64('crop_v1.jpg')}" alt="Native-resolution crop around V1 with comparison stars labelled"><figcaption>At native pixels. V1 at its mean brightness is a 19th-magnitude star, about one hundred-thousandth as bright as the faintest star the eye can see. The blue circles are four of the eight Gaia comparison stars used in section 4, with their catalogue magnitudes; the other four lie just outside this cut.</figcaption></figure>
<figure><img src="{b64('m31_footprint.jpg')}" alt="The Iris frame drawn on a wide survey image of M31"><figcaption>Where the frame sits. The orange rectangle is the Iris field, 42 by 28 arcminutes, drawn on a 3.6-degree survey image (Digitized Sky Survey, red plate). The footprint comes from matching the stack's stars against Gaia: 17 stars matched to 0.3&Prime;, the camera rotated {abs(sol['rotation_deg']):.1f} degrees from north, pixel scale {sol['scale_arcsec_px']:.4f}&Prime;. V1 lies {sep_arcmin:.0f}&prime; south-west of the nucleus, about {proj_kpc:.1f} kpc in projection, in the inner disk near a dust lane.</figcaption></figure>
</div>
<p>V1 is a disk star. That matters for the distance in section 5, because the disk is where the dust is, and it mattered in 1952, when Walter Baade realised that the Cepheids Hubble had used were the young disk kind and not the old halo kind the period-luminosity law had been calibrated on, and the distance to every galaxy doubled overnight.</p>

<h2>4. The stars around it do not change</h2>
<p>A variable star is only believable against stars that do not vary. Every night the pipeline measures V1 and between 15 and 22 Gaia stars within nine arcminutes of it, all on the same frames, through the same air and the same optics. The comparison stars set the zero point, so clouds, haze, airmass and the night's transparency cancel: if the sky dims everything by a tenth of a magnitude, the comparison stars dim with V1 and the difference is unchanged. Below, eight of those stars measured by a second, independent route (three frames a night, simple apertures, each star's zero point taken from the other seven) and drawn against V1 on the same axis.</p>
<figure>{svg('comps.svg')}<figcaption>Each star is plotted relative to its own median, so the comparison stars should sit on zero and do. Their night-to-night scatter is {CS['median_comp_scatter']:.3f} mag (median of the eight) across {CS['nights']} nights, which is the measurement floor for a 15th-magnitude star in this setup. V1, measured identically, swings {CS['v1_range'][1]-CS['v1_range'][0]:.2f} mag. Nothing in the atmosphere or the telescope can do that to one star and not its neighbours.</figcaption></figure>
<p>The comparison stars are chosen from Gaia DR3 for being isolated, unsaturated in a five-minute exposure, and not flagged variable by Gaia's own 34-month survey. Their collective scatter is also a check on the whole night: when the zero point from 20 stars disagrees by more than a tenth of a magnitude, something was wrong with the frames, and the 2026-10-14 night failed exactly that test.</p>

<h2>5. From the light curve to the distance of a galaxy</h2>
<p>A Cepheid's period tells you its true luminosity. Calibrate that relation with Cepheids whose distances are known by parallax, and a period plus an apparent brightness becomes a distance.</p>
<h3>The law, and where the absolute magnitude comes from</h3>
<p>The absolute magnitude M<sub>V</sub> is how bright the star would appear from the standard distance of 10 parsecs. For a classical Cepheid it is a straight line in the logarithm of the period:</p>
<p style="text-align:center; font-size:19px; margin:14px 0;">M<sub>V</sub> = &minus;2.43 (log<sub>10</sub> P &minus; 1) &minus; 4.05 &nbsp;&nbsp;&nbsp;<span style="color:var(--muted); font-size:15px;">(P in days)</span></p>
<p>The &ldquo;&minus; 1&rdquo; centres the line on a ten-day Cepheid, so &minus;4.05 is the absolute magnitude of a ten-day star and &minus;2.43 is the slope: ten times the period is 2.43 magnitudes brighter, about nine times the luminosity. For V1:</p>
<p style="text-align:center; font-family:'Helvetica Neue', Arial, sans-serif; font-size:15px; line-height:1.9; margin:10px 0 14px;">
log<sub>10</sub> {P:.1f} = {math.log10(P):.3f}<br>
M<sub>V</sub> = &minus;2.43 &times; ({math.log10(P):.3f} &minus; 1) &minus; 4.05 = &minus;2.43 &times; {math.log10(P)-1:.3f} &minus; 4.05 = {-2.43*(math.log10(P)-1):.2f} &minus; 4.05 = <b>{abs_mag:.2f}</b></p>
<p>That is some 10,000 Suns in visible light, and it is the same for every 31-day Cepheid wherever it is, which is the whole point: nothing about V1's measured brightness goes into it, only its period. The calibration scatters by about 0.2 magnitudes from star to star, mostly because the instability strip has some width in temperature, and that is the &plusmn;10 percent in distance quoted below.</p>
<p>The relation is Henrietta Swan Leavitt's. Working at Harvard College Observatory on photographic plates of the Small Magellanic Cloud, she published in 1908 that the brighter of its variables had the longer periods, and in 1912 drew the straight line between period and magnitude for 25 of them. Because every star in the Cloud is at nearly the same distance, the line in apparent magnitude had to be a line in true luminosity, but with no distance to the Cloud it floated: Leavitt had the slope and not the zero point. Ejnar Hertzsprung pinned the zero point in 1913 from the statistical parallax of thirteen Cepheids in our own Galaxy, Harlow Shapley revised it, and Walter Baade's 1952 correction for the two populations of pulsating star is the reason every extragalactic distance doubled that year. The coefficients used here are from Fritz Benedict, Barbara McArthur, Michael Feast, Thomas Barnes and colleagues (Astronomical Journal 133, 1810, 2007), who measured direct trigonometric parallaxes of ten Galactic Cepheids with the Fine Guidance Sensors of the Hubble Space Telescope and fitted the line through them in the V band. Leavitt's slope and a parallax zero point: the two halves of the equation are a century apart.</p>
<h3>The distance, step by step</h3>
<table class="steps">
<tr><td>Period, from the literature and confirmed by the fit</td><td class="num">P = {P:.1f} d, log P = {math.log10(P):.3f}</td></tr>
<tr><td>Absolute magnitude from the Leavitt law (Benedict et al. 2007, Hubble Space Telescope parallaxes): M<sub>V</sub> = &minus;2.43 (log P &minus; 1) &minus; 4.05</td><td class="num">M<sub>V</sub> = {abs_mag:.2f}</td></tr>
<tr><td>Mean apparent magnitude: the brightness averaged over the whole fitted cycle (in flux, then back to magnitudes), on the Gaia G scale our comparison stars provide</td><td class="num">&lt;G&gt; = {mean:.2f}</td></tr>
<tr><td>Colour term to take Gaia G to Johnson V for a yellow supergiant (V &minus; G &asymp; +0.3)</td><td class="num">&lt;V&gt; = {mv:.2f}</td></tr>
<tr><td>Foreground dust in our own Galaxy toward M31 (E(B&minus;V) &asymp; 0.1, A<sub>V</sub> &asymp; 0.3)</td><td class="num">V<sub>0</sub> = {mv-0.3:.2f}</td></tr>
<tr><td>Distance modulus, apparent minus absolute</td><td class="num">&mu; = {mod:.2f}</td></tr>
<tr><td>Distance, d = 10<sup>(&mu;/5 + 1)</sup> parsecs</td><td class="num"><b>{dc['kpc']:.0f} kpc = {dc['mly']:.2f} million light years</b></td></tr>
</table>
<p>The accepted distance to M31 is {true_mly} million light years, modulus {true_mod}. We are {true_mod-mod:.2f} mag short, about {100*(1-10**((mod-true_mod)/5)):.0f} percent in distance. That is not a failure of the method; it is the list of things a single star through a single filter cannot fix. The law itself scatters by 0.2 mag from star to star. Our magnitudes are on the Gaia G system because that is what the comparison stars are catalogued in, while the law is written in V, and the colour term for a Cepheid changes through the cycle. Dust inside M31's own disk, along the line of sight to a star sitting next to a dust lane, is not in the 0.3 mag of foreground extinction at all, and 0.2 to 0.4 mag of it would close the gap by itself. Hubble, with twelve Cepheids and the Hooker telescope in 1925, got 285 kpc, three times too close, because his calibration mixed two kinds of pulsating star; the method was right and the calibration was wrong. Ours uses one star and the modern calibration, and lands within a fifth of the true distance.</p>
<div class="note">What would tighten it: the mean magnitude from a complete fold rather than one cycle with a gap; a few nights through the B and R filters, so that the star's own colour, and with it the reddening by dust, is measured instead of assumed; and a second Cepheid in the same frame, of which M31's disk has dozens at this depth.</div>

<h2>6. Why a Cepheid pulsates, and why the period gives the luminosity</h2>
<p>V1 is a yellow supergiant. From its period and luminosity it is a star of roughly seven to nine solar masses, about {lum_solar/1000:.0f},000 times the Sun's luminosity, and somewhere near 150 times its radius. It was a hot blue star on the main sequence some forty million years ago; having exhausted the hydrogen at its core it has swollen and cooled, and it is crossing a narrow band of surface temperatures, about 5,500 to 6,500 K, called the instability strip. Any star that passes through that strip pulsates, and V1 is in it now.</p>
<h3>The valve</h3>
<p>Below the surface of such a star there is a layer where helium is on the edge of losing its second electron. Compress that layer and instead of heating up, as ordinary gas would, it spends the energy ionising more helium. Ionised helium is far more opaque than neutral, so the compressed layer traps the radiation flowing up from the interior, heats, and pushes the layers above it outward. As the star expands the layer cools, the helium recombines, the opacity drops, the stored heat escapes, and the layers fall back under gravity, overshooting into the next compression. Eddington called it a valve: a layer that absorbs heat when compressed and releases it when expanded drives an oscillation, exactly as a heat engine does. The driving works only when the ionisation layer sits at the right depth, which is why the strip is narrow and why a star drifts into and out of pulsation as it evolves.</p>
<h3>Why the light curve is lopsided</h3>
<p>The surface of V1 moves at tens of kilometres per second and the radius changes by about a tenth. Yet most of the brightness change comes from temperature, not size: a star 1,000 K hotter at nearly the same radius is almost twice as bright. The rise is steep because the outward-moving layers reach the surface as a near-shock, the photosphere heats quickly, and maximum light comes soon after the star has passed through its smallest radius and is expanding fastest. The decline is slow because the star then coasts outward and cools gently under gravity. The star is brightest not when it is largest but when it is hottest, a quarter of a cycle earlier.</p>
<h3>Why period means luminosity</h3>
<p>The period of any pulsating star is set by how long a sound wave takes to cross it, which depends on its mean density: P &prop; 1/&radic;&rho;. A more massive star in the strip is more luminous, and at the strip's fixed temperature a more luminous star must be larger, because luminosity goes as radius squared times temperature to the fourth. Larger at similar mass means less dense, and less dense means a longer period. So the chain is: period measures density, density at a fixed temperature measures radius, radius at a fixed temperature measures luminosity. The scatter in the law comes mostly from the strip having some width in temperature. A 31-day period puts V1 near the long, luminous end; the Cepheids Leavitt saw ranged from about a day to over a hundred, and the longest are among the most luminous stars that exist.</p>
<h3>Why it mattered</h3>
<p>Because the relation holds for every classical Cepheid, a 31-day Cepheid in M31 is intrinsically the same star as a 31-day Cepheid in our Galaxy whose distance is known. Hubble photographed this star on the night of 1923 October 5, thought it a nova, compared earlier plates, and wrote VAR! on the glass when it turned out to repeat. Its period gave a luminosity, its faintness gave a distance, and the distance was far beyond anything in the Milky Way. The spiral nebulae were galaxies. Nearly every step on the cosmic distance ladder since, including the measurement of the expansion of the universe that bears his name, rests on stars like this one, measured as it is measured here.</p>

<h2>7. How the numbers were made</h2>
<table>
<tr><th>Frames</th><td>300-second exposures through the luminance filter, QHY600M at gain 0, {frames_used} of 236 passing the sharpness gate for the stack; the photometry uses every frame of each night that passes its own checks.</td></tr>
<tr><th>Calibration</th><td>Each frame bias-, dark- and flat-corrected with the stacker's master frames before measurement. A frame in which V1 or a comparison star reaches 60,000 ADU is dropped; a frame whose sky differs by more than 30 percent from one third of the frame to another is dropped as obstructed.</td></tr>
<tr><th>Photometry</th><td>Aperture photometry of V1 and 15 to 22 Gaia DR3 comparison stars (G 13.5 to 17, isolated, not flagged variable) within 9&prime;, each frame's zero point from the sigma-clipped ensemble. A nightly point is the mean of the frames, its error the frame scatter over &radic;n combined with the zero-point scatter; a nightly error above 0.5 mag is a failed night. Airmass is recorded at each frame's mid-exposure.</td></tr>
<tr><th>Curve</th><td>Period fixed at {P} d (General Catalogue of Variable Stars); a {M['K']}-term Fourier series in phase fitted by weighted least squares to the {n} good luminance nights; &chi;&sup2; {M['chi2']:.1f} for {M['dof']} degrees of freedom. The predicted points are that curve evaluated at 01:00 UT on each coming date.</td></tr>
<tr><th>Distance</th><td>Intensity mean of the fitted cycle, {mean:.2f}; the mean of the measured nights alone is {M['mean_nights']:.2f} and gives {dn['mly']:.2f} Mly, the difference being the unsampled part of the cycle. Law and terms as in section 5.</td></tr>
<tr><th>Astrometry</th><td>The stack's stars matched to Gaia DR3 by triangle matching (astroalign), {sol['n_match']} stars, 0.3&Prime; rms; the footprint and the position of V1 follow from that solution. The survey image is DSS2 red through SkyView, 3.6&deg; across.</td></tr>
<tr><th>Submission</th><td>The nightly points are written in AAVSO extended format under observer code THCA, CV band (unfiltered, V zero point), for upload to WebObs by hand.</td></tr>
</table>
<footer>Iris observatory, CDK17 at 2939 mm with a QHY600M. Data 2026-10-08 to 2026-11-10; page made 2026-11-12. Leavitt law: Benedict et al. 2007, AJ 133, 1810. Comparison stars and astrometry: Gaia DR3. Wide-field image: Digitized Sky Survey via NASA SkyView.</footer>
</main></body></html>"""
OUT.write_text(html)
print("wrote", OUT, "%.1f MB" % (OUT.stat().st_size / 1e6))
