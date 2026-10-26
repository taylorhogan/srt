// Iris desk display: M5Stack Core2 v1.3 (2026-10-22).
//
// Polls GET /api/dash on the chat server every 30 s and paints it. Display
// only: touch and the buttons do nothing, and the payload carries no command.
// Pages rotate (2026-10-25): status 20 s, then 10 s each of the night's FWHM
// chart, the newest frame (GET /api/dash?part=image, fetched only when the
// frame changes) and last night's summary -- each only when there is data.
// The colors are decided by the server (cmd_processing/dash.py): g fine,
// a look at it, r act now, x no data. If no good reply has arrived for
// STALE_MS the whole screen goes grey with the age, so an old "SHUT" is never
// read as current.
//
// Build: Arduino IDE or arduino-cli, board "M5Core2" (M5Stack board package),
// libraries M5Unified and ArduinoJson 7. Copy secrets.h.example to secrets.h.
// M5Unified, not the old M5Core2 library: v1.3 has the AXP2101 power chip.

#include <M5Unified.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "secrets.h"   // WIFI_SSID, WIFI_PASS, DASH_URL
#include <Update.h>
#include "mbedtls/md.h"
#include "esp_ota_ops.h"
// Over-the-air update (2026-10-26). Both headers are written by
// scripts/core2_publish.py and gitignored; a build without them still runs,
// it just never updates itself.
#if __has_include("fw_version.h")
#include "fw_version.h"
#else
#define FW_VERSION "usb"
#endif
#if __has_include("ota_key.h")
#include "ota_key.h"
#define HAVE_OTA 1
#else
#define HAVE_OTA 0
#endif

static const uint32_t POLL_MS = 30000;
static const uint32_t STALE_MS = 180000;
static const uint8_t BRIGHTNESS = 90;          // 0-255
// Portrait: 0 or 2. If the picture is upside down for the way the unit
// stands (USB-C at the bottom), switch to the other one.
static const uint8_t ROTATION = 2;

static JsonDocument doc;
static bool haveData = false;

// While imaging (doc["im"] set) only the three big-type pages rotate: time and
// temperature, the target's counts per filter, its median/last FWHM per filter
// (owner 2026-10-26: "simpler information while imaging"). Otherwise the
// status / FWHM / picture / summary pages as before.
enum Page { PG_STATUS, PG_FWHM, PG_IMAGE, PG_SUMMARY, PG_BIG_TIME, PG_BIG_COUNT, PG_BIG_FWHM, PG_NPAGES };
static const uint32_t PAGE_MS[PG_NPAGES] = {20000, 10000, 10000, 10000, 10000, 7000, 7000};
static long serverSec = -1;                    // seconds since local midnight at the last good reply
static int page = PG_STATUS;
static uint32_t pageStart = 0;

static uint8_t* jpg = nullptr;                 // newest frame thumbnail, in PSRAM
static size_t jpgLen = 0;
static const size_t JPG_MAX = 64 * 1024;
static String jpgId = "";
static uint32_t lastOk = 0, lastPoll = 0;
static String lastErr = "starting";

// Wi-Fi on a mesh (2026-10-25): by default the ESP32 joins the FIRST access
// point that answers and never leaves it, so in the bedroom it held a distant
// node and stopped updating until carried outside. Connect to the strongest
// node, and rescan when the link is weak or two polls in a row fail -- at most
// every 2 minutes, so it cannot thrash.
static String failedFw = "";      // a version that failed its check: not tried again
static bool markedValid = false;
static int failsInRow = 0;
static uint32_t lastRejoin = 0;
static const int WEAK_RSSI = -78;               // dBm
static const uint32_t REJOIN_MS = 120000;

static void joinStrongest() {
  WiFi.disconnect();
  WiFi.setScanMethod(WIFI_ALL_CHANNEL_SCAN);
  WiFi.setSortMethod(WIFI_CONNECT_AP_BY_SIGNAL);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  lastRejoin = millis();
}

static void checkLink() {
  if (millis() - lastRejoin < REJOIN_MS) return;
  bool weak = WiFi.status() == WL_CONNECTED && WiFi.RSSI() < WEAK_RSSI;
  if (failsInRow >= 2 || weak || WiFi.status() != WL_CONNECTED) {
    joinStrongest();
    failsInRow = 0;
  }
}

static uint16_t colorFor(const char* c) {
  if (!c) return TFT_DARKGREY;
  switch (c[0]) {
    case 'g': return TFT_GREEN;
    case 'a': return TFT_ORANGE;
    case 'r': return TFT_RED;
    case 'w': return TFT_WHITE;
    default:  return TFT_DARKGREY;
  }
}

static void fetch() {
  if (WiFi.status() != WL_CONNECTED) { lastErr = "no wifi"; return; }
  HTTPClient http;
  http.setTimeout(15000);                      // first call after idle can take ~5 s
  if (!http.begin(DASH_URL)) { lastErr = "bad url"; return; }
  int code = http.GET();
  if (code == 200) {
    JsonDocument fresh;
    DeserializationError e = deserializeJson(fresh, http.getStream());
    if (!e && fresh["ok"].as<bool>()) {
      doc = fresh; haveData = true; lastOk = millis(); lastErr = ""; failsInRow = 0;
      serverSec = doc["sec"].isNull() ? -1 : (long)doc["sec"];
      if (!markedValid) { esp_ota_mark_app_valid_cancel_rollback(); markedValid = true; }
    } else {
      lastErr = e ? "bad json" : "server not ok"; failsInRow++;
    }
  } else {
    lastErr = "http " + String(code); failsInRow++;
  }
  http.end();
}


// Over-the-air update. The server advertises "fw": {"v", "size", "hmac"}.
// When v differs from this build, the image is streamed into the spare app
// slot while HMAC-SHA256 (key in ota_key.h) is computed over it, and the
// boot slot is switched ONLY if the HMAC matches -- so a corrupted or
// tampered download is discarded and this firmware keeps running.
static void otaScreen(const char* msg, int pct) {
  auto& d = M5.Display;
  d.startWrite();
  d.fillScreen(TFT_BLACK);
  d.fillRect(0, 0, 240, 30, TFT_DARKGREY);
  d.setTextColor(TFT_BLACK, TFT_DARKGREY);
  d.setTextSize(2); d.setCursor(8, 8); d.print("IRIS");
  d.setTextColor(TFT_WHITE, TFT_BLACK);
  d.setCursor(8, 120); d.print(msg);
  if (pct >= 0) {
    d.drawRect(8, 150, 224, 16, TFT_LIGHTGREY);
    d.fillRect(10, 152, 220 * pct / 100, 12, TFT_CYAN);
  }
  d.endWrite();
}

static void checkFirmware() {
#if HAVE_OTA
  if (!haveData || doc["fw"].isNull() || WiFi.status() != WL_CONNECTED) return;
  String v = doc["fw"]["v"] | "";
  String want = doc["fw"]["hmac"] | "";
  int size = doc["fw"]["size"] | 0;
  if (v.length() == 0 || v == FW_VERSION || v == failedFw) return;
  if (size <= 0 || want.length() != 64) return;
  HTTPClient http;
  http.setTimeout(20000);
  if (!http.begin(String(DASH_URL) + "?part=firmware")) return;
  if (http.GET() != 200 || http.getSize() != size) { http.end(); return; }
  if (!Update.begin(size)) { http.end(); failedFw = v; return; }
  otaScreen("updating", 0);
  mbedtls_md_context_t ctx;
  mbedtls_md_init(&ctx);
  mbedtls_md_setup(&ctx, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 1);
  mbedtls_md_hmac_starts(&ctx, OTA_KEY, sizeof(OTA_KEY));
  WiFiClient* st = http.getStreamPtr();
  static uint8_t buf[2048];
  size_t got = 0;
  int lastPct = -1;
  bool writeOk = true;
  uint32_t t0 = millis();
  while (got < (size_t)size && millis() - t0 < 180000) {
    size_t want_n = (size_t)size - got;
    if (want_n > sizeof(buf)) want_n = sizeof(buf);
    int n = st->readBytes(buf, want_n);
    if (n <= 0) { delay(5); continue; }
    mbedtls_md_hmac_update(&ctx, buf, n);
    if (Update.write(buf, n) != (size_t)n) { writeOk = false; break; }
    got += n;
    int pct = (int)(100.0 * got / size);
    if (pct / 5 != lastPct / 5) { otaScreen("updating", pct); lastPct = pct; }
  }
  uint8_t mac[32];
  mbedtls_md_hmac_finish(&ctx, mac);
  mbedtls_md_free(&ctx);
  http.end();
  char hex[65];
  for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", mac[i]);
  if (writeOk && got == (size_t)size && want.equalsIgnoreCase(hex) && Update.end(true)) {
    otaScreen("restarting", 100);
    delay(800);
    ESP.restart();
  }
  Update.abort();
  failedFw = v;                                   // keep running this build
  otaScreen("update refused", -1);
  delay(3000);
#endif
}

// The thumbnail, only when the newest frame has changed since the last one.
static void fetchImage() {
  if (!haveData || doc["img"].isNull() || WiFi.status() != WL_CONNECTED) return;
  String id = doc["img"]["id"] | "";
  if (id == jpgId) return;
  if (!jpg) jpg = (uint8_t*)ps_malloc(JPG_MAX);
  if (!jpg) return;
  HTTPClient http;
  http.setTimeout(15000);
  if (!http.begin(String(DASH_URL) + "?part=image")) return;
  if (http.GET() == 200) {
    int len = http.getSize();
    if (len > 0 && (size_t)len <= JPG_MAX) {
      WiFiClient* st = http.getStreamPtr();
      size_t got = 0;
      uint32_t t0 = millis();
      while (got < (size_t)len && millis() - t0 < 10000) {
        int n = st->readBytes(jpg + got, len - got);
        if (n <= 0) delay(5); else got += n;
      }
      if (got == (size_t)len) { jpgLen = got; jpgId = id; }
    }
  }
  http.end();
}

// Portrait (2026-10-24): the Core2 stands on its side so the USB-C cable
// leaves from the bottom. 240 x 320, so each row is a small grey label with
// its value BELOW it, full width. The built-in font is 6 px wide per text
// size: a value of up to 13 characters is size 3, longer drops to size 2
// (20 characters) rather than running off the right edge.
static const int W = 240;

static void row(int y, const char* label, const String& value, uint16_t color) {
  auto& d = M5.Display;
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setTextSize(1);
  d.setCursor(8, y);
  d.print(label);
  d.setTextColor(color, TFT_BLACK);
  bool big = value.length() * 18 <= W - 12;
  d.setTextSize(big ? 3 : 2);
  d.setCursor(8, big ? y + 11 : y + 14);
  d.print(value.substring(0, 20));
}

static void paintStatus() {
  auto& d = M5.Display;
  d.startWrite();
  d.fillScreen(TFT_BLACK);
  bool stale = !haveData || millis() - lastOk > STALE_MS;

  // Top band: overall level and the clock; the night state on the line below.
  uint16_t band = stale ? TFT_DARKGREY : colorFor(doc["level"]);
  d.fillRect(0, 0, W, 30, band);
  d.setTextColor(TFT_BLACK, band);
  d.setTextSize(2);
  d.setCursor(8, 8);
  d.print("IRIS");
  if (haveData) {
    d.setCursor(W - 12 * 5 - 8, 8);
    d.print(doc["t"].as<const char*>());
  }

  if (!haveData) {
    row(60, "STATUS", "NO DATA", TFT_DARKGREY);
    d.setTextSize(2); d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.setCursor(8, 110); d.print(lastErr);
    // What the router needs for a DHCP reservation, and what the chat's
    // allowlist will be given: shown until the first good reply.
    d.setTextSize(1);
    d.setCursor(8, 150); d.print("IP  "); d.print(WiFi.localIP().toString());
    d.setCursor(8, 166); d.print("MAC "); d.print(WiFi.macAddress());
    d.setCursor(8, 182); d.print("URL "); d.print(DASH_URL);
    d.setCursor(8, 198); d.printf("RSSI %d dBm", WiFi.status() == WL_CONNECTED ? (int)WiFi.RSSI() : 0);
    d.setCursor(8, 214); d.print("FW  "); d.print(FW_VERSION);
    d.endWrite();
    return;
  }

  auto grey = [&](uint16_t c) { return stale ? (uint16_t)TFT_DARKGREY : c; };

  d.setTextSize(2);
  d.setTextColor(grey(colorFor(doc["night"]["c"])), TFT_BLACK);
  d.setCursor(8, 38);
  d.print(doc["night"]["v"] | "");

  // Roof (north camera's newest read) with its age.
  String roof = doc["roof"]["v"] | "?";
  if (!doc["roof"]["age"].isNull()) {
    int age = doc["roof"]["age"];
    roof += age < 60 ? "  " + String(age) + "m" : "  " + String(age / 60) + "h";
  }
  row(64, "ROOF", roof, grey(colorFor(doc["roof"]["c"])));
  row(108, "SCOPE", doc["scope"]["v"] | "?", grey(colorFor(doc["scope"]["c"])));

  // Now imaging, or what comes next.
  if (!doc["now"].isNull()) {
    String now = String((const char*)doc["now"]["dso"]) + " " + (const char*)doc["now"]["filter"]
                 + " " + String((int)doc["now"]["n"]);
    if (!doc["now"]["of"].isNull()) now += "/" + String((int)doc["now"]["of"]);
    row(152, "NOW", now, grey(TFT_CYAN));
  } else if (!doc["next"].isNull()) {
    row(152, "NEXT", String((const char*)doc["next"]["dso"]) + " " + (const char*)doc["next"]["at"],
        grey(TFT_WHITE));
  } else {
    row(152, "NEXT", doc["tonight"]["image"] ? "-" : "no imaging", grey(TFT_WHITE));
  }

  // Roof window and disk, one per line.
  d.setTextSize(2);
  d.setTextColor(grey(TFT_LIGHTGREY), TFT_BLACK);
  const char* op = doc["tonight"]["opens"] | "--:--";
  const char* cl = doc["tonight"]["closes"] | "--:--";
  d.setCursor(8, 202); d.printf("open  %s", op);
  d.setCursor(8, 224); d.printf("close %s", cl);
  if (!doc["disk"].isNull()) {
    d.setTextColor(grey(colorFor(doc["disk"]["c"])), TFT_BLACK);
    d.setCursor(8, 246);
    d.printf("disk  %d GB", (int)doc["disk"]["gb"]);
  }
  // The sky camera: stars seen as a share of a clear sky (white under a moon,
  // which hides stars without any cloud).
  if (!doc["sky"].isNull()) {
    d.setTextColor(grey(colorFor(doc["sky"]["c"])), TFT_BLACK);
    d.setCursor(8, 268);
    d.printf("sky   %s%s", (const char*)(doc["sky"]["v"] | "?"), doc["sky"]["moon"] ? " moon" : "");
  }

  // Bottom: alerts, or how old the data is when it has gone stale.
  if (stale) {
    d.setTextColor(TFT_ORANGE, TFT_BLACK);
    d.setCursor(8, 282);
    d.printf("NO UPDATE %lus", (unsigned long)((millis() - lastOk) / 1000));
    d.setTextSize(1);
    d.setCursor(8, 304);
    d.printf("%s  wifi %d dBm", lastErr.c_str(), WiFi.status() == WL_CONNECTED ? (int)WiFi.RSSI() : 0);
  } else if (doc["alerts"].size() > 0) {
    d.setTextColor(colorFor(doc["level"]), TFT_BLACK);
    d.setCursor(8, 290);
    d.print(String(doc["alerts"][0].as<const char*>()).substring(0, 19));
  }
  d.endWrite();
}

// The band every page shares: overall level and the clock.
static void band(const char* title) {
  auto& d = M5.Display;
  uint16_t b = colorFor(doc["level"]);
  d.fillRect(0, 0, W, 30, b);
  d.setTextColor(TFT_BLACK, b);
  d.setTextSize(2);
  d.setCursor(8, 8);
  d.print(title);
  d.setCursor(W - 12 * 5 - 8, 8);
  d.print(doc["t"].as<const char*>());
}

static uint16_t filterColor(const char* f) {
  String s = f ? f : "";
  if (s == "L") return TFT_WHITE;
  if (s == "R" || s == "Ha") return TFT_RED;
  if (s == "G") return TFT_GREEN;
  if (s == "B") return 0x4C9F;                  // light blue
  if (s == "O-III") return TFT_CYAN;
  if (s == "S-II") return TFT_ORANGE;
  return TFT_LIGHTGREY;
}

// Page 2: one dot per frame, colored by filter; thin lines where a slot
// changes target; the dashed median; the newest value large at the bottom.
static void paintFwhm() {
  auto& d = M5.Display;
  JsonObject f = doc["fwhm"];
  band("FWHM");
  d.setTextSize(2);
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setCursor(8, 38);
  d.print(doc["summary"].isNull() ? "tonight" : "last night");
  const int X0 = 30, X1 = 234, Y0 = 76, Y1 = 250;
  int span = f["span"] | 60;
  float lo = 1.5, hi = 3.5;
  for (JsonArray p : f["p"].as<JsonArray>()) hi = max(hi, (int)p[1] / 100.0f + 0.1f);
  auto X = [&](int m) { return X0 + (X1 - X0) * m / max(span, 1); };
  auto Y = [&](float v) { v = constrain(v, lo, hi); return (int)(Y1 - (Y1 - Y0) * (v - lo) / (hi - lo)); };
  d.setTextSize(1);
  for (float v = lo; v <= hi + 0.01f; v += 0.5f) {
    d.drawFastHLine(X0, Y(v), X1 - X0, 0x2945);
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.setCursor(2, Y(v) - 3); d.printf("%.1f", v);
  }
  // Hour ticks from the start time.
  const char* st = f["start"] | "00:00";
  int sm = atoi(st) * 60 + atoi(st + 3);
  for (int m = 60 - sm % 60; m <= span; m += 60) {
    d.drawFastVLine(X(m), Y1, 3, TFT_LIGHTGREY);
    d.setCursor(X(m) - 5, Y1 + 6); d.printf("%02d", ((sm + m) / 60) % 24);
  }
  // Slot changes, names staggered so neighbours do not overprint.
  int lastEnd = -99, rowOff = 0;
  for (JsonArray sl : f["slots"].as<JsonArray>()) {
    int x = X((int)sl[0]);
    String name = String((const char*)sl[1]).substring(0, 8);
    if (sl[0].as<int>() > 0) d.drawFastVLine(x - 2, Y0, Y1 - Y0, 0x5ACB);
    rowOff = (x < lastEnd + 2) ? 1 - rowOff : 0;
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.setCursor(x, Y0 - 14 + 9 * rowOff); d.print(name);
    lastEnd = x + 6 * name.length();
  }
  int med = f["med"] | 0;
  for (int x = X0; x < X1; x += 6) d.drawFastHLine(x, Y(med / 100.0f), 3, TFT_DARKGREY);
  JsonArray fl = f["f"];
  for (JsonArray p : f["p"].as<JsonArray>())
    d.fillCircle(X((int)p[0]), Y((int)p[1] / 100.0f), 2, filterColor(fl[(int)p[2]] | "?"));
  d.setTextSize(2);
  d.setTextColor(filterColor(f["last"][1] | "?"), TFT_BLACK);
  d.setCursor(8, 270);
  d.printf("%.2f\" %s", (int)f["last"][0] / 100.0f, (const char*)(f["last"][1] | "?"));
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setCursor(8, 294); d.printf("med %.2f\" n=%d", med / 100.0f, (int)f["p"].size());
}

// Page 3: the newest frame, centre third of the field, as the server
// rendered it.
static void paintImage() {
  auto& d = M5.Display;
  band("LAST");
  d.setTextSize(2);
  d.setTextColor(filterColor(doc["img"]["filter"] | "?"), TFT_BLACK);
  d.setCursor(8, 40);
  d.printf("%s %s", (const char*)(doc["img"]["dso"] | "?"), (const char*)(doc["img"]["filter"] | "?"));
  d.drawJpg(jpg, jpgLen, 0, 72, 240, 160);
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setCursor(8, 246);
  if (!doc["img"]["fwhm"].isNull()) d.printf("FWHM %.2f\"", doc["img"]["fwhm"].as<float>());
  d.setCursor(8, 270); d.printf("at %s", (const char*)(doc["img"]["at"] | ""));
}

// Page 4: what last night brought, target by target.
static void paintSummary() {
  auto& d = M5.Display;
  JsonObject s = doc["summary"];
  band("NIGHT");
  d.setTextSize(2);
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setCursor(8, 38); d.print(s["night"].as<const char*>() + 5);   // MM-DD
  d.setCursor(W - 12 * 6 - 8, 38); d.printf("%3d fr", (int)s["total"]);
  int y = 66;
  for (JsonObject t : s["targets"].as<JsonArray>()) {
    d.setTextSize(2);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    d.setCursor(8, y); d.print(t["dso"].as<const char*>());
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.setCursor(W - 12 * 4 - 8, y); d.printf("%.2f", t["fwhm"].as<float>());
    d.setTextSize(1);
    d.setCursor(8, y + 18);
    for (JsonPair kv : t["n"].as<JsonObject>()) {
      d.setTextColor(filterColor(kv.key().c_str()), TFT_BLACK);
      d.printf("%s %d  ", kv.key().c_str(), kv.value().as<int>());
    }
    y += 36;
  }
  d.setTextSize(2);
  d.setTextColor(TFT_GREEN, TFT_BLACK);
  d.setCursor(8, 262); d.printf("best %.2f\"", s["best"]["fwhm"].as<float>());
  d.setTextSize(1);
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setCursor(8, 286);
  d.printf("%s %s at %s", (const char*)s["best"]["dso"], (const char*)s["best"]["filter"],
           (const char*)s["best"]["at"]);
}


// ---- big-type imaging pages ------------------------------------------------
// Text is scaled to fill its box (M5GFX scales any font), so the numbers are
// as large as the 240 x 320 screen allows.
static void fitText(const String& txt, int cx, int cy, int maxW, int maxH,
                    const lgfx::IFont* font, uint16_t color) {
  auto& d = M5.Display;
  d.setFont(font);
  d.setTextSize(1);
  float w = d.textWidth(txt), h = d.fontHeight();
  if (w < 1 || h < 1) return;
  float k = min((float)maxW / w, (float)maxH / h);
  d.setTextSize(k);
  d.setTextDatum(middle_center);
  d.setTextColor(color, TFT_BLACK);
  d.drawString(txt, cx, cy);
}

static void plainFont() {
  auto& d = M5.Display;
  d.setFont(&fonts::Font0);
  d.setTextSize(1);
  d.setTextDatum(top_left);
}

static String clockNow() {
  if (serverSec < 0) return String(doc["t"] | "--:--");
  long sec = (serverSec + (long)((millis() - lastOk) / 1000)) % 86400;
  char b[6];
  snprintf(b, sizeof(b), "%02ld:%02ld", sec / 3600, (sec / 60) % 60);
  return String(b);
}

// Page A: the time and the outside temperature, each half the screen.
static void paintBigTime() {
  auto& d = M5.Display;
  d.fillRect(0, 0, 240, 6, colorFor(doc["level"]));          // the level, as a thin bar
  fitText(clockNow(), 120, 85, 228, 140, &fonts::Font7, TFT_WHITE);
  String temp = doc["temp_f"].isNull() ? String("--") : String((int)round(doc["temp_f"].as<float>()));
  fitText(temp + "F", 120, 238, 228, 140, &fonts::FreeSansBold24pt7b, TFT_CYAN);
  plainFont();
}

// Page B: the target, then one line per filter: "Ha: 12".
static void paintBigCount() {
  JsonObject im = doc["im"];
  JsonArray fs = im["filters"];
  int n = max(1, (int)fs.size());
  fitText(String((const char*)(im["dso"] | "?")), 120, 30, 228, 52, &fonts::FreeSansBold24pt7b, TFT_WHITE);
  int top = 64, rowH = min(64, (320 - top) / n);
  int i = 0;
  for (JsonObject f : fs) {
    String line = String((const char*)f["f"]) + ": " + String((int)f["n"]);
    fitText(line, 120, top + rowH * i + rowH / 2, 228, rowH - 6, &fonts::FreeSansBold24pt7b,
            filterColor(f["f"] | "?"));
    i++;
  }
  plainFont();
}

// Page C: the same filters, median and last FWHM: "Ha 1.84 1.79".
static void paintBigFwhm() {
  JsonObject im = doc["im"];
  JsonArray fs = im["filters"];
  int n = max(1, (int)fs.size());
  fitText("FWHM med last", 120, 30, 228, 40, &fonts::FreeSansBold24pt7b, TFT_LIGHTGREY);
  int top = 64, rowH = min(64, (320 - top) / n);
  int i = 0;
  for (JsonObject f : fs) {
    char b[32];
    snprintf(b, sizeof(b), "%s %.2f %.2f", (const char*)f["f"], f["med"].as<float>(), f["last"].as<float>());
    fitText(String(b), 120, top + rowH * i + rowH / 2, 228, rowH - 6, &fonts::FreeSansBold24pt7b,
            filterColor(f["f"] | "?"));
    i++;
  }
  plainFont();
}

static bool imaging() { return haveData && !doc["im"].isNull(); }

static bool pageHasData(int p) {
  bool im = imaging();
  if (p == PG_BIG_TIME) return im;
  if (p == PG_BIG_COUNT || p == PG_BIG_FWHM) return im && doc["im"]["filters"].size() > 0;
  if (im) return false;                         // imaging: only the big pages
  switch (p) {
    case PG_STATUS:  return true;
    case PG_FWHM:    return !doc["fwhm"].isNull();
    case PG_IMAGE:   return jpgLen > 0 && !doc["img"].isNull();
    case PG_SUMMARY: return !doc["summary"].isNull();
  }
  return false;
}

// Stale data, or none yet, holds the status page: it is the one that says so.
static void paint() {
  bool stale = !haveData || millis() - lastOk > STALE_MS;
  if (stale) page = PG_STATUS;
  else if (!pageHasData(page)) page = imaging() ? PG_BIG_TIME : PG_STATUS;
  if (page == PG_STATUS) { paintStatus(); return; }
  M5.Display.startWrite();
  M5.Display.fillScreen(TFT_BLACK);
  if (page == PG_FWHM) paintFwhm();
  else if (page == PG_IMAGE) paintImage();
  else if (page == PG_SUMMARY) paintSummary();
  else if (page == PG_BIG_TIME) paintBigTime();
  else if (page == PG_BIG_COUNT) paintBigCount();
  else paintBigFwhm();
  M5.Display.endWrite();
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(ROTATION);
  M5.Display.setBrightness(BRIGHTNESS);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  joinStrongest();
  paint();
}

void loop() {
  M5.update();                                 // battery/power housekeeping only
  bool repaint = false;
  if (lastPoll == 0 || millis() - lastPoll >= POLL_MS) {
    lastPoll = millis();
    checkLink();
    fetch();
    checkFirmware();                           // before anything else uses the reply
    fetchImage();
    repaint = (page == PG_STATUS);                // other pages refresh on their next turn
  }
  if (millis() - pageStart >= PAGE_MS[page]) {
    int next = page;
    for (int k = 0; k < PG_NPAGES; k++) {        // the next page that has something to show
      next = (next + 1) % PG_NPAGES;
      if (pageHasData(next)) break;
    }
    page = next;
    pageStart = millis();
    repaint = true;
  }
  // The big clock follows the minute between polls.
  static String shownClock = "";
  if (page == PG_BIG_TIME && clockNow() != shownClock) { shownClock = clockNow(); repaint = true; }
  if (repaint) paint();
  delay(200);
}
