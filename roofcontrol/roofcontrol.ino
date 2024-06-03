
// todo figure out open/close wiring and default stat
// todo add in i2c display of door state, action, and incoming commands

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

String serialin;  //incoming serial data
String str;       //store the state  of opened/closed/safe pins

//relays
#define toggle_direction 4
#define sensor 5


//input sensors
#define opened 11  // roof open  sensor
#define closed 12  // roof closed sensor
#define safe 13    // scope  safety sensor

// display stuff
#define SCREEN_WIDTH 128     // OLED display width, in pixels
#define SCREEN_HEIGHT 32     // OLED display height, in pixels
#define SCREEN_ADDRESS 0x3C  ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32
// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET -1  // Reset pin # (or -1 if sharing Arduino reset pin)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


int simStatus = 0;



unsigned long end_time;
bool lost = false;  //roof not reporting state

void setup() 
{
  
  end_time = millis() + 60000;  //roof lost reset timer ~60 seconds  change to suit  your rquirments to determine if roof is lost

  //Begin Serial Comunication(configured  for 9600baud)
  Serial.begin(9600);

  //pin relay as OUTPUT
  pinMode(toggle_direction, OUTPUT);




  //pins as INPUT
  pinMode(closed, INPUT_PULLUP);
  pinMode(opened, INPUT_PULLUP);
  pinMode(safe, INPUT_PULLUP);



  Serial.write("RRCI#");  //init string
 
}

void loop() {




  //Verify connection by serial

  while (Serial.available() > 0) {
    Serial.println("waiting");
    //Read  Serial data and alocate on serialin
    serialin = Serial.readStringUntil('#');
    Serial.println(serialin);

    if (serialin == "on") {  // turn scope sensor on
      digitalWrite(sensor, HIGH);
    }

    if (serialin == "off") {  // turn scope sensor off
      digitalWrite(sensor, LOW);
    }

    if (serialin == "x") {
      digitalWrite(toggle_direction, LOW);

    }

    else if (serialin == "open") {
      simStatus = 2;

      simStatus = 1;
      digitalWrite(toggle_direction, HIGH);
      delay(1000);
      digitalWrite(toggle_direction, LOW);

    }


    else if (serialin == "close") {

      simStatus = 3;

      simStatus = 0;
      digitalWrite(toggle_direction, HIGH);
      delay(1000);
      digitalWrite(toggle_direction, LOW);
    }



    if (serialin == "Parkstatus") {  // exteranl query command to fetch RRCI data

      Serial.println("0#");
      serialin = "";
    }

    if (serialin == "get") {

      if (simStatus == 0) {
        str += "closed,safe,";
      } else if (simStatus == 1) {
        str += "open,safe,";
      } else {
        str += "unknown,safe,";
      }




      if (simStatus == 1 && (lost == false)) {
        str += "not_moving_o#";
        end_time = millis() + 60000;  //reset the timer
      }

      if (simStatus == 0 && (lost == false)) {
        str += "not_moving_c#";
        end_time = millis() + 60000;  //reset the timer
      }
      if (simStatus == 4 && (lost == false)) {
        str += "moving#";
      }

      Serial.println(str);  //send serial data
      serialin = "";
      str = "";
      //delay(100);
    }
  }
  serialin = "";
}
