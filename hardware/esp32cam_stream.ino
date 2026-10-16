/**
 * ESP32-CAM MJPEG Stream
 * Tuoi toi uu cho pipeline optical flow / nhan biet nhip tho:
 *   - Do phan giai: VGA (640x480) - du cho YOLO pose + optical flow,
 *     khong qua nang cho WiFi o toc do on dinh.
 *   - JPEG quality: 12 (thang 0-63, so CANG NHO CANG TOT).
 *     Quality 12 giam artifact block so voi mac dinh (10), van giu
 *     bang thong hop ly. Optical flow Farneback rat nhay cam voi
 *     JPEG block artifact -- day la tham so quan trong nhat de chinh.
 *   - Framerate thuc te: ~10-15 fps qua WiFi 2.4GHz (du cho bandpass
 *     0.1-0.5 Hz vi can toi thieu 1 Hz = 2x tan so cao nhat).
 *
 * Phan cung: AI Thinker ESP32-CAM (cam bien OV2640).
 * Ket noi: GPIO0 voi GND khi nap code, roi go bo sau khi nap xong.
 *
 * Thu vien can: "ESP32" board package cua Espressif trong Arduino IDE.
 * (Tools -> Board -> Boards Manager -> tim "esp32" -> Install)
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ---- CAU HINH WIFI ---- (doi thanh thong tin mang cua ban) ----
const char* WIFI_SSID     = "TEN_WIFI_CUA_BAN";
const char* WIFI_PASSWORD = "MAT_KHAU_WIFI";

// ---- Chan GPIO cho AI Thinker ESP32-CAM ----
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

// ---------------------------------------------------------------

httpd_handle_t stream_httpd = NULL;

// Handler tra ve MJPEG stream lien tuc
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t* fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len;
  uint8_t* _jpg_buf;
  char part_buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if (res != ESP_OK) return res;

  // Ngan client (breathing_worker.py) bi timeout khi FPS thap
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    _jpg_buf_len = fb->len;
    _jpg_buf = fb->buf;

    // Gui MIME boundary
    size_t hlen = snprintf(part_buf, sizeof(part_buf),
      "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
      _jpg_buf_len);

    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)_jpg_buf, _jpg_buf_len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, "\r\n", 2);

    esp_camera_fb_return(fb);

    if (res != ESP_OK) break; // client ngat ket noi

    // Gioi han ~12 fps de giam tai WiFi va tranh buffer day.
    // breathing_worker.py tu tinh fps thuc te tu timestamp, nen
    // khong can chinh xac tuyet doi; on dinh quan trong hon la nhanh.
    delay(80); // ~12.5 fps toi da
  }
  return res;
}

// Handler don gian de kiem tra cam co hoat dong khong (GET /ping)
static esp_err_t ping_handler(httpd_req_t *req) {
  httpd_resp_send(req, "OK", 2);
  return ESP_OK;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };
  httpd_uri_t ping_uri = {
    .uri       = "/ping",
    .method    = HTTP_GET,
    .handler   = ping_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &ping_uri);
    Serial.println("Camera server started");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Booting...");

  // --- Khoi tao camera ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = CAM_PIN_D0;
  config.pin_d1       = CAM_PIN_D1;
  config.pin_d2       = CAM_PIN_D2;
  config.pin_d3       = CAM_PIN_D3;
  config.pin_d4       = CAM_PIN_D4;
  config.pin_d5       = CAM_PIN_D5;
  config.pin_d6       = CAM_PIN_D6;
  config.pin_d7       = CAM_PIN_D7;
  config.pin_xclk     = CAM_PIN_XCLK;
  config.pin_pclk     = CAM_PIN_PCLK;
  config.pin_vsync    = CAM_PIN_VSYNC;
  config.pin_href     = CAM_PIN_HREF;
  config.pin_sscb_sda = CAM_PIN_SIOD;
  config.pin_sscb_scl = CAM_PIN_SIOC;
  config.pin_pwdn     = CAM_PIN_PWDN;
  config.pin_reset    = CAM_PIN_RESET;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // VGA (640x480): du cho YOLO pose detect + optical flow.
  // SVGA (800x600) hoac UXGA (1600x1200) se lam giam FPS,
  // QVGA (320x240) se lam giam do chinh xac YOLO.
  config.frame_size   = FRAMESIZE_VGA;

  // Chat luong JPEG: 12 (thang 0-63, cang nho cang tot ve chat luong).
  // Khong nen de duoi 10 (qua nhieu artifact) hoac tren 20 (tang byte
  // qua nhieu, WiFi qua tai, FPS giam manh).
  config.jpeg_quality = 12;

  // 2 frame buffer: trong khi 1 frame dang duoc gui qua WiFi, cam
  // co the chup frame tiep theo vao buffer thu 2 -> giam jitter FPS.
  config.fb_count     = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init FAILED: 0x%x\n", err);
    // Khoi dong lai sau 5 giay de tu phuc hoi neu camera bi loi tam thoi
    delay(5000);
    ESP.restart();
    return;
  }

  // Tinh chinh cam bien OV2640 de giam nhieu -> optical flow chinh xac hon
  sensor_t* s = esp_camera_sensor_get();
  s->set_brightness(s, 0);       // 0 = trung tinh
  s->set_contrast(s, 1);         // Tuong phan nhe giup optical flow bam tot hon
  s->set_saturation(s, -1);      // Giam bao hoa -> tiep can anh xam, phu hop hon
  s->set_sharpness(s, 1);        // Sac net nhe giup Farneback bam edge tot hon
  s->set_denoise(s, 1);          // Bat khu nhieu on sensor (co tren OV2640)
  s->set_whitebal(s, 1);         // Bat can bang trang tu dong
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1);    // Bat do sang tu dong
  s->set_aec2(s, 1);             // AEC nang cao
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);        // Bat AGC (tu dong dieu chinh gain)
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)2); // Gioi han gain, giam hat nhieu

  // --- Ket noi WiFi ---
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false); // Tat WiFi power save -> giam do tre, on dinh hon

  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi FAILED - restarting");
    delay(3000);
    ESP.restart();
    return;
  }

  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  Serial.println("Stream: http://" + WiFi.localIP().toString() + "/stream");
  Serial.println("Ping:   http://" + WiFi.localIP().toString() + "/ping");

  startCameraServer();
}

void loop() {
  // Kiem tra WiFi dinh ky, tu khoi dong lai neu mat ket noi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost - reconnecting...");
    WiFi.reconnect();
    delay(5000);
  }
  delay(1000);
}
