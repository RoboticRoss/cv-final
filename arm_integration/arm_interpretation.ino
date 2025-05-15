#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  120
#define SERVOMAX  600

// CODE FOR THE ARDUINO MEGA!!!!
// This processes serial data sent from the main 

// ~~~~ Servo Channels ~~~~
const int BASE_CHANNEL     = 0;
const int SHOULDER_CHANNEL = 4;
const int ELBOW_CHANNEL    = 8;
const int CLAW_CHANNEL    = 12;

// ~~~~ Joint Limits ~~~~
const int BASE_MIN      = 5;
const int BASE_MAX      = 155;

const int SHOULDER_MIN  = 5; //fully out
const int SHOULDER_MAX  = 85; //fully up

const int ELBOW_MIN     = 40;
const int ELBOW_MAX     = 160;

const int CLAW_MIN     = 5;
const int CLAW_MAX     = 55;



// ~~~~ Default Angles ~~~~
float baseAngle     = 90;
float shoulderAngle = 80;
float elbowAngle    = 100;
float clawAngle     = 30;

// ~~~~ Target Angles ~~~~
float baseTarget     = 90;
float shoulderTarget = 80;
float elbowTarget    = 100;
float clawTarget     = 30;


const float alpha = 0.07;  // 0.03 works well for slow movments, use .07 in demo

int angleToPulse(float angle) {
  return map((int)angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Kinda yandaredev code but kept this for fine tuning
void moveBase(int target) {
  if (target < BASE_MIN || target > BASE_MAX) return;
  baseAngle = (1 - alpha) * baseAngle + alpha * target;
  pwm.setPWM(BASE_CHANNEL, 0, angleToPulse(baseAngle));
}

void moveShoulder(int target) {
  if (target < SHOULDER_MIN || target > SHOULDER_MAX) return;
  shoulderAngle = (1 - alpha) * shoulderAngle + alpha * target;
  pwm.setPWM(SHOULDER_CHANNEL, 0, angleToPulse(shoulderAngle));
}

void moveElbow(int target) {
  if (target < ELBOW_MIN || target > ELBOW_MAX) return;
  elbowAngle = (1 - alpha) * elbowAngle + alpha * target;
  pwm.setPWM(ELBOW_CHANNEL, 0, angleToPulse(elbowAngle));
}

void moveClaw(int target) {
  if (target < CLAW_MIN || target > CLAW_MAX) return;
  clawAngle = (1 - alpha) * clawAngle + alpha * target;
  pwm.setPWM(CLAW_CHANNEL, 0, angleToPulse(clawAngle));
}

bool isValidPositiveInteger(const char* str) {
  if (str == NULL || *str == '\0') return false;
  for (const char* ptr = str; *ptr != '\0'; ++ptr) {
    if (!isdigit(*ptr)) return false;
  }
  return true;
}

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(50);

  delay(1000);
}


void loop() {
  static String inputString = "";
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      //check for "START"
      int angles[4];
      int index = 0;

      char inputCopy[inputString.length() + 1];
      inputString.toCharArray(inputCopy, sizeof(inputCopy));

      char *token = strtok(inputCopy, ",");
      if (token && String(token) == "START") { //Basically a start tag
        token = strtok(NULL, ",");
        while (token != NULL && index < 4) {
          int val = atoi(token);
          if (val > 0) {
            angles[index++] = val;
          } else {
            break; 
          }
          token = strtok(NULL, ",");
        }

        if (index == 4) {
          baseTarget     = constrain(angles[0], BASE_MIN, BASE_MAX);
          shoulderTarget = constrain(angles[1], SHOULDER_MIN, SHOULDER_MAX);
          elbowTarget    = constrain(angles[2], ELBOW_MIN, ELBOW_MAX);
          clawTarget     = constrain(angles[3], CLAW_MIN, CLAW_MAX);
        }
      }

      inputString = "";
    } else {
      inputString += inChar;
    }
  }
  moveBase(baseTarget);
  moveShoulder(shoulderTarget);
  moveElbow(elbowTarget);
  moveClaw(clawTarget);

  delay(20);
  
}

void sweepJoint(int channel, int startDeg, int endDeg, const char* label) {
  int step = (startDeg < endDeg) ? 1 : -1;

  for (int angle = startDeg; angle != endDeg + step; angle += step) {
    int pulse = angleToPulse(angle);
    pwm.setPWM(channel, 0, pulse);

    Serial.print(label);
    Serial.print(" Angle: ");
    Serial.println(angle);

    delay(100);  // super slow
  }
  //SWEEEOEOPPPPS
  //sweepJoint(BASE_CHANNEL, BASE_MIN, BASE_MAX, "Base");
  //sweepJoint(SHOULDER_CHANNEL, SHOULDER_MIN, SHOULDER_MAX, "Shoulder");
  //sweepJoint(ELBOW_CHANNEL, ELBOW_MIN, ELBOW_MAX, "Elbow");
  //sweepJoint(CLAW_CHANNEL, CLAW_MIN, CLAW_MAX, "Claw");
  //sweepJoint(CLAW_CHANNEL, CLAW_MAX, CLAW_MIN, "Claw");
  //sweepJoint(ELBOW_CHANNEL, ELBOW_MAX, ELBOW_MIN, "Elbow");
  //sweepJoint(SHOULDER_CHANNEL, SHOULDER_MAX, SHOULDER_MIN, "Shoulder");
  //sweepJoint(BASE_CHANNEL, BASE_MAX, BASE_MIN, "Base");
}
