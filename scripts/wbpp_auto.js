// wbpp_auto.js — Headless PixInsight WBPP automation for Iris observatory
//
// Usage (cron / command line):
//   /Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight \
//     --no-interface --run=/path/to/wbpp_auto.js \
//     /path/to/calDir /path/to/outputDir
//
// Directory layout expected:
//   <targetsDir>/
//     <calDir>/          e.g. "cdk"
//       <YYYY-MM-DD>/
//         FLAT/  DARK/  BIAS/
//     <DSO1>/
//       <YYYY-MM-DD>/
//         LIGHT/
//     <DSO2>/ ...
//
// The script finds the DSO with the most recent imaging date,
// collects ALL light frames for that DSO (across all dates),
// pairs them with ALL calibration frames, and runs WBPP.

// Frame type constants for WBPP inputFrames
var FRAME_LIGHT = 0;
var FRAME_BIAS  = 1;
var FRAME_DARK  = 2;
var FRAME_FLAT  = 3;

// ── helpers ──────────────────────────────────────────────────────────────────

function listSubdirs(dirPath) {
    var result = [];
    if (!File.directoryExists(dirPath)) return result;
    var entries = File.directoryEntries(dirPath, File.FullPath | File.Directories);
    for (var i = 0; i < entries.length; ++i) {
        var name = File.extractName(entries[i]);
        if (name !== "." && name !== "..") result.push(name);
    }
    result.sort();
    return result;
}

function findFitsFiles(dirPath) {
    var results = [];
    if (!File.directoryExists(dirPath)) return results;
    var entries = File.directoryEntries(dirPath, File.FullPath | File.Files | File.Directories);
    for (var i = 0; i < entries.length; ++i) {
        var entry = entries[i];
        if (File.directoryExists(entry)) {
            var name = File.extractName(entry);
            if (name !== "." && name !== "..") {
                results = results.concat(findFitsFiles(entry));
            }
        } else {
            var ext = File.extractExtension(entry).toLowerCase();
            if (ext === ".fits" || ext === ".fit" || ext === ".xisf") {
                results.push(entry);
            }
        }
    }
    return results;
}

// Walk calDir/<date>/<typeDir>/ and collect all FITS files.
function collectCalFrames(calDir, typeDir) {
    var results = [];
    var dateDirs = listSubdirs(calDir);
    for (var i = 0; i < dateDirs.length; ++i) {
        var typePath = calDir + "/" + dateDirs[i] + "/" + typeDir;
        results = results.concat(findFitsFiles(typePath));
    }
    return results;
}

// Return {name, dir} of the DSO whose most-recent date subdir is newest.
function mostRecentDso(targetsDir, calName) {
    var dsoDirs = listSubdirs(targetsDir);
    var bestDso = null;
    var bestDate = "";
    for (var i = 0; i < dsoDirs.length; ++i) {
        var name = dsoDirs[i];
        if (name === calName) continue;
        var dsoPath = targetsDir + "/" + name;
        var dateDirs = listSubdirs(dsoPath);
        if (dateDirs.length === 0) continue;
        var newest = dateDirs[dateDirs.length - 1]; // sorted, so last = newest
        if (newest > bestDate) {
            bestDate = newest;
            bestDso = {name: name, dir: dsoPath, date: newest};
        }
    }
    return bestDso;
}

// Collect ALL light frames across all date subdirs for a DSO.
function collectLightFrames(dsoDir) {
    var results = [];
    var dateDirs = listSubdirs(dsoDir);
    for (var i = 0; i < dateDirs.length; ++i) {
        var lightPath = dsoDir + "/" + dateDirs[i] + "/LIGHT";
        results = results.concat(findFitsFiles(lightPath));
    }
    return results;
}

// Build WBPP inputFrames row: [enabled, path, frameType, filter, binX, binY, expTime, temp]
// Filter/binning/exposure/temp are read from FITS headers by WBPP, so leave as defaults.
function makeFrame(path, frameType) {
    return [true, path, frameType, "", 1, 1, 0.0, 0.0];
}

// ── main ─────────────────────────────────────────────────────────────────────

function main() {
    // Parse arguments — PixInsight passes script args via jsArguments[]
    var calDir    = (typeof jsArguments !== "undefined" && jsArguments.length > 0) ? jsArguments[0] : "";
    var outputDir = (typeof jsArguments !== "undefined" && jsArguments.length > 1) ? jsArguments[1] : "";

    if (!calDir || !File.directoryExists(calDir)) {
        console.criticalln("ERROR: calDir not specified or does not exist: " + calDir);
        console.criticalln("Usage: wbpp_auto.js <calDir> <outputDir>");
        return;
    }
    if (!outputDir) {
        console.criticalln("ERROR: outputDir not specified.");
        console.criticalln("Usage: wbpp_auto.js <calDir> <outputDir>");
        return;
    }
    if (!File.directoryExists(outputDir)) {
        console.noteln("Output directory does not exist, creating: " + outputDir);
        File.createDirectory(outputDir, true);
    }

    // Derive targetsDir (parent of calDir) and calName
    var calDir     = calDir.replace(/\/+$/, "");    // strip trailing slash
    var calName    = File.extractName(calDir);
    var targetsDir = File.extractDrive(calDir) + File.extractDirectory(calDir).replace(/\/+$/, "");

    console.noteln("=== WBPP Auto ===");
    console.noteln("Targets dir : " + targetsDir);
    console.noteln("Cal dir     : " + calDir + " (" + calName + ")");
    console.noteln("Output dir  : " + outputDir);

    // Discover most-recent DSO
    var dso = mostRecentDso(targetsDir, calName);
    if (!dso) {
        console.criticalln("ERROR: No DSO directories found under " + targetsDir);
        return;
    }
    console.noteln("Target DSO  : " + dso.name + " (most recent date: " + dso.date + ")");

    // Collect frames
    var lights = collectLightFrames(dso.dir);
    var flats  = collectCalFrames(calDir, "FLAT");
    var darks  = collectCalFrames(calDir, "DARK");
    var bias   = collectCalFrames(calDir, "BIAS");

    console.noteln("Light frames: " + lights.length);
    console.noteln("Flat frames : " + flats.length);
    console.noteln("Dark frames : " + darks.length);
    console.noteln("Bias frames : " + bias.length);

    if (lights.length === 0) {
        console.criticalln("ERROR: No light frames found for " + dso.name);
        return;
    }
    if (flats.length === 0) console.warningln("WARNING: No flat frames found");
    if (darks.length === 0) console.warningln("WARNING: No dark frames found");
    if (bias.length  === 0) console.warningln("WARNING: No bias frames found");

    // Build WBPP inputFrames array
    var inputFrames = [];
    for (var i = 0; i < lights.length; ++i) inputFrames.push(makeFrame(lights[i], FRAME_LIGHT));
    for (var i = 0; i < flats.length;  ++i) inputFrames.push(makeFrame(flats[i],  FRAME_FLAT));
    for (var i = 0; i < darks.length;  ++i) inputFrames.push(makeFrame(darks[i],  FRAME_DARK));
    for (var i = 0; i < bias.length;   ++i) inputFrames.push(makeFrame(bias[i],   FRAME_BIAS));

    console.noteln("Total frames: " + inputFrames.length);

    // ── Run WBPP ─────────────────────────────────────────────────────────────
    // WBPP is a script-based process. In headless mode it may not be available
    // by default. If it throws, see the fallback comment block below.
    try {
        var P = new WeightedBatchPreprocessing;
        P.inputFrames    = inputFrames;
        P.outputDirectory = outputDir;
        console.noteln("Starting WBPP...");
        P.executeGlobal();
        console.noteln("WBPP complete. Output: " + outputDir);
    } catch (e) {
        console.criticalln("ERROR: Could not run WeightedBatchPreprocessing: " + e.message);
        console.criticalln("WBPP may not be available in headless mode on your PixInsight version.");
        console.criticalln("");
        console.criticalln("Fallback option: use native process modules (ImageCalibration +");
        console.criticalln("StarAlignment + ImageIntegration). See inline comments in this script.");
        //
        // ── FALLBACK (enable if WBPP unavailable headlessly) ─────────────────
        // Uncomment and implement the three steps below if needed:
        //
        // STEP 1 — Calibrate lights
        // var IC = new ImageCalibration;
        // IC.inputFrames = lights;
        // IC.masterBiasEnabled = bias.length > 0;
        // IC.masterBiasPath    = bias.length > 0 ? bias[0] : "";   // or a pre-made master
        // IC.masterDarkEnabled = darks.length > 0;
        // IC.masterDarkPath    = darks.length > 0 ? darks[0] : "";
        // IC.masterFlatEnabled = flats.length > 0;
        // IC.masterFlatPath    = flats.length > 0 ? flats[0] : "";
        // IC.outputDirectory   = outputDir + "/calibrated";
        // IC.executeGlobal();
        //
        // STEP 2 — Register (StarAlignment)
        // var SA = new StarAlignment;
        // SA.referenceImage  = calibratedFiles[0];
        // SA.inputFrames     = calibratedFiles;
        // SA.outputDirectory = outputDir + "/registered";
        // SA.executeGlobal();
        //
        // STEP 3 — Integrate (ImageIntegration)
        // var II = new ImageIntegration;
        // II.inputFrames     = registeredFiles;
        // II.outputDirectory = outputDir;
        // II.executeGlobal();
    }
}

main();
