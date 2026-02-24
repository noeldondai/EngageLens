# EngageLens (AI Classroom Engagement + Weekly Report System)

## Overview
EngageLens is an AI-powered classroom support tool that helps teachers and parents understand student engagement trends.
It uses computer vision (webcam) to detect basic behavior states (e.g., attentive vs distracted like phone-use / head-down),
logs the time spent in each state, and (idea) combines it with attendance + assignment/test info to generate weekly reports.

## Problem
Teachers can’t track every student’s engagement all the time, and parents often get updates only when it’s too late.
Manual reporting is time-consuming and inconsistent.

## Our Solution
- Real-time webcam-based engagement detection (prototype)
- Timer-based logging per “state” (attentive / phone / head-down / etc.)
- CSV logs per session/day
- Weekly report generation (planned) by combining:
  - attention logs + attendance + assignment submissions + assessments
- Teacher-controlled features (can enable/disable phone detection, posture tracking, or AI entirely)

## Features (What Works in This Repo)
- Live webcam monitoring scripts (YOLO-based phone detection + attention timers)
- CSV/log generation for engagement time tracking
- Prototype app/dashboard (if applicable)

## Planned / Future Work
- Merge logs with school platform data (attendance + submissions + exams)
- Auto-generate weekly PDF report with charts
- Parent notification system (email/WhatsApp/SMS integration)

## Tech Stack
- Python
- OpenCV
- YOLO (Ultralytics)
- (Optional) Flask + templates (if using `app.py` / `templates/`)

## Project Structure
- `app.py` - main app entry (if dashboard is used)
- `templates/` - UI templates
- `webcam_*.py` - real-time detection + timers
- `train_no_val.py` - training script (if used)
- `logs/` / `artifacts/` - generated outputs (not pushed to GitHub)

## Setup
### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
