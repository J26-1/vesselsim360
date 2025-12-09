# VesselSim360 – Smart Marine Safety & Training System

VesselSim360 is a **web-based marine safety platform** designed for small vessel crews. It combines **IoT-based incident detection**, **AI-generated safety guidance**, and **3D interactive training simulations** to improve maritime situational awareness and emergency preparedness.

The system works both **online and offline**, making it suitable for remote sea environments.

---

## 🚢 System Description

VesselSim360 enhances marine safety by integrating three core capabilities:

1. **Detect incidents** in real time using simulated IoT sensor data  
2. **Provide immediate, localized emergency guidance** using an AI hybrid model  
3. **Generate interactive 3D training scenarios** so the crew can learn from incidents  

It is designed for small vessels such as fishing boats, tour boats, sport boats, or recreational craft.

---

## 📦 Key System Components

### 1. Incident Detection Module
Monitors live or simulated IoT sensor feeds to detect marine safety hazards.

- Supports common maritime sensors:
  - Tilt / pitch / roll sensors (capsizing risk)
  - GPS drift (anchor drag or unexpected movement)
  - Engine status (failure, overheat)
  - Power/battery indicators
- Uses threshold rules + safety logic
- Produces event triggers for guidance and simulation modules

---

### 2. AI Safety Guidance
A hybrid **rules-based + lightweight NLP model** capable of running offline.

- Provides localized step-by-step emergency instructions  
- Works without internet (edge-friendly)  
- Contains:
  - Maritime safety rule set
  - Context-aware AI explanations

**Examples of guidance:**
- “Hull tilt is increasing beyond safe angle. Move weight to starboard and reduce throttle.”  
- “GPS drift detected. Anchor may be dragging. Check seabed and extend chain length.”

---

### 3. 3D Simulation Viewer
Transforms incident data into **interactive 3D training scenes**.

- Built using **Three.js / WebGL**
- Renders:
  - Point cloud environments
  - Vessel models (OBJ/GLB)
- Allows crews to:
  - Replay incidents  
  - Review vessel motion  
  - Interact with camera and controls  

---

### 4. Performance Dashboard
Evaluates crew response and training quality.

- Measures:
  - Reaction time  
  - Correctness of procedural steps  
  - Decision-making  
- Shows:
  - Strengths  
  - Weaknesses  
  - Recommended drills  

---

## 🏗️ System Architecture Overview

Simulated IoT Sensors
          ↓
Incident Detection Engine
          ↓
AI Safety Guidance (Offline)
          ↓
3D Scenario Generator → Three.js Viewer
          ↓
Performance Dashboard → User Analytics

---


## 🛠️ Technologies Used

### Frontend
- React.js / JavaScript  
- Three.js (3D rendering)  
- TailwindCSS (UI)

### Backend
- Python or Node.js  
- Rule-based detection engine  
- Offline NLP inference

### Data
- Sensor event logs  
- 3D models  
- Performance and training history  

---

## 🚀 Features

- ✔ Real-time sensor-based incident detection  
- ✔ Offline-capable AI emergency assistant  
- ✔ Auto-generated 3D training scenarios  
- ✔ Interactive WebGL simulation viewer  
- ✔ Performance analysis and scoring  
- ✔ Modular design for integration and extension  


