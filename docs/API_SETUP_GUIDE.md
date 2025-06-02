# 🛰️ Space Debris API Setup Guide

Complete guide to accessing real-time space debris tracking APIs for enhanced data collection.

---

## 📋 **API OVERVIEW & COMPARISON**

| API | Cost | Objects | Update Freq | Registration | Difficulty |
|-----|------|---------|-------------|--------------|------------|
| **CelesTrak** | 🟢 FREE | 25,000+ | 30 seconds | ❌ None | ⭐ Easy |
| **Space-Track.org** | 🟢 FREE | 40,000+ | Multiple/day | ✅ Required | ⭐⭐ Moderate |
| **ESA Space Debris** | 🟢 FREE | 15,000+ | Daily | ✅ Required | ⭐⭐ Moderate |
| **LeoLabs** | 🔴 PAID | 20,000+ | Real-time | ✅ Commercial | ⭐⭐⭐ Advanced |
| **N2YO** | 🟡 FREEMIUM | 5,000+ | Real-time | ✅ API Key | ⭐⭐ Moderate |

---

## 🚀 **1. CelesTrak TLE Data** ⭐ **FASTEST & EASIEST**

### **✅ Advantages:**
- 🆓 **Completely FREE** - No registration required
- ⚡ **Fastest updates** - Every 30 seconds
- 🔄 **Multiple formats** - TLE, JSON, XML, CSV
- 🛰️ **25,000+ objects** - All active satellites + debris
- 🌍 **Global coverage** - Worldwide tracking

### **📝 Setup Instructions:**
```
⚡ INSTANT ACCESS - NO SETUP REQUIRED!
```

### **🔗 API Endpoints:**
```
# All Active Satellites (TLE Format)
https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle

# All Active Satellites (JSON Format)  
https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json

# Space Debris Only
https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=json

# Last 30 Days Launches
https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=json

# All Starlink Satellites
https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json
```

### **⚡ Usage Example:**
```python
import requests

# Get all active satellites in JSON format
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
response = requests.get(url)
satellites = response.json()

print(f"Retrieved {len(satellites)} satellites")
```

### **📊 Rate Limits:**
- ✅ **No rate limits** for reasonable use
- 🔄 **Update every 30 seconds** for real-time data
- 📱 **No authentication** required

---

## 🏛️ **2. Space-Track.org (18th SDS)** ⭐ **MOST COMPREHENSIVE**

### **✅ Advantages:**
- 🆓 **FREE** with registration
- 📊 **40,000+ objects** - Most comprehensive database
- 🛰️ **Official U.S. data** - Department of Defense
- 🔍 **Advanced filtering** - Query by date, object type, etc.
- 📈 **Historical data** available

### **📝 Setup Instructions:**

#### **Step 1: Account Registration**
1. 🌐 Go to: https://www.space-track.org/auth/createAccount
2. 📧 **Email:** Use a valid email address
3. 👤 **Personal Info:** Fill out required fields
4. 🏢 **Organization:** Can be "Personal Research" or your institution
5. 📞 **Phone:** Required for verification
6. ⏳ **Wait:** Account approval can take 24-48 hours

#### **Step 2: Account Verification**
1. 📧 Check email for verification link
2. 🔍 May require additional documentation
3. 📞 Possible phone verification call
4. ✅ Account approved → Login credentials ready

#### **Step 3: API Access**
```python
# Login credentials (keep secure!)
USERNAME = "your_space_track_username"
PASSWORD = "your_space_track_password"
```

### **🔗 API Endpoints:**
```
# Login (required for each session)
https://www.space-track.org/ajaxauth/login

# All Satellites
https://www.space-track.org/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID/format/json

# Recent TLE Data
https://www.space-track.org/basicspacedata/query/class/tle_latest/orderby/NORAD_CAT_ID/format/json

# Debris Only
https://www.space-track.org/basicspacedata/query/class/satcat/OBJECT_TYPE/DEBRIS/format/json

# Last 30 Days
https://www.space-track.org/basicspacedata/query/class/satcat/LAUNCH/>now-30/format/json
```

### **⚡ Usage Example:**
```python
import requests

# Login session
session = requests.Session()
login_data = {
    'identity': 'your_username',
    'password': 'your_password'
}

# Authenticate
session.post('https://www.space-track.org/ajaxauth/login', data=login_data)

# Get satellite data
url = 'https://www.space-track.org/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID/limit/1000/format/json'
response = session.get(url)
satellites = response.json()
```

### **📊 Rate Limits:**
- 📈 **200 requests per hour** (free account)
- ⏰ **30 requests per minute** maximum
- 💰 **Premium accounts** available for higher limits

---

## 🇪🇺 **3. ESA Space Debris Office API**

### **✅ Advantages:**
- 🆓 **FREE** for research use
- 🛰️ **15,000+ objects** - European focus
- 📊 **Debris-specific data** - Specialized in space junk
- 🔬 **Research quality** - High accuracy data
- 🌍 **European coverage** - Strong European satellite tracking

### **📝 Setup Instructions:**

#### **Step 1: Registration**
1. 🌐 Go to: https://sdup.esoc.esa.int/sdup/
2. 📧 Create account with research institution email (preferred)
3. 📝 Fill research purpose and intended use
4. ⏳ Wait for approval (1-3 business days)

#### **Step 2: API Access Request**
1. 📧 Email: space.debris@esa.int
2. 📝 Include: Research purpose, institution, data usage plan
3. 📋 Request API access credentials
4. ⏳ Approval process: 1-2 weeks

### **🔗 API Endpoints:**
```
# Base URL (credentials required)
https://sdup.esoc.esa.int/sdup/api/

# Object catalog
/objects

# TLE data
/tle

# Conjunction assessments
/conjunctions
```

### **📊 Rate Limits:**
- 📈 **1000 requests per day** (research account)
- ⏰ **Updates daily** - Not real-time
- 🔍 **Batch queries** recommended

---

## 💰 **4. LeoLabs API** ⭐ **PREMIUM REAL-TIME**

### **✅ Advantages:**
- ⚡ **Real-time tracking** - Sub-second updates
- 🎯 **Highest accuracy** - Commercial grade precision
- 🛰️ **20,000+ objects** - Including small debris
- 📊 **Advanced analytics** - Collision predictions
- 🔮 **Future predictions** - Orbital projections

### **📝 Setup Instructions:**

#### **Step 1: Commercial Contact**
1. 🌐 Go to: https://www.leolabs.space/
2. 📞 Contact sales: sales@leolabs.space
3. 📝 Explain use case and requirements
4. 💰 Discuss pricing (typically $$$$ per month)

#### **Step 2: Trial Access**
- 🆓 **Free trial** sometimes available
- 📊 **Limited data** during trial
- ⏰ **Time-limited** access

### **💰 Pricing:**
- 🔬 **Research:** $500-2000/month
- 🏢 **Commercial:** $5000+/month
- 🎓 **Academic discounts** sometimes available

---

## 📡 **5. N2YO Satellite Tracking API**

### **✅ Advantages:**
- 🟡 **Freemium model** - Basic free tier
- ⚡ **Real-time positions** - Current satellite locations
- 🌍 **Visual tracking** - Pass predictions
- 📱 **Easy integration** - Simple REST API

### **📝 Setup Instructions:**

#### **Step 1: Registration**
1. 🌐 Go to: https://www.n2yo.com/api/
2. 📧 Sign up for free account
3. 🔑 Get API key immediately

#### **Step 2: API Key**
```python
API_KEY = "your_n2yo_api_key"
```

### **🔗 API Endpoints:**
```
# Get satellite positions
https://api.n2yo.com/rest/v1/satellite/positions/{id}/{observer_lat}/{observer_lng}/{observer_alt}/{seconds}&apiKey={api_key}

# Above observer
https://api.n2yo.com/rest/v1/satellite/above/{observer_lat}/{observer_lng}/{observer_alt}/{search_radius}/{category_id}&apiKey={api_key}

# TLE data
https://api.n2yo.com/rest/v1/satellite/tle/{id}&apiKey={api_key}
```

### **📊 Rate Limits:**
- 🆓 **Free:** 1000 requests/hour
- 💰 **Paid:** Up to 100,000 requests/hour

---

## 🔧 **AUTHENTICATION SETUP CHECKLIST**

### **For Your Implementation:**

#### **✅ Required Credentials:**
```python
# config.py or environment variables
SPACE_TRACK_USERNAME = "your_username"
SPACE_TRACK_PASSWORD = "your_password"
ESA_API_KEY = "your_esa_key"  # If approved
LEOLABS_API_KEY = "your_leolabs_key"  # If subscribed
N2YO_API_KEY = "your_n2yo_key"
```

#### **✅ Recommended Priority:**
1. 🥇 **Start with CelesTrak** (immediate, no setup)
2. 🥈 **Register for Space-Track.org** (most comprehensive)
3. 🥉 **Sign up for N2YO** (real-time positions)
4. 🎯 **Consider ESA** (if research focused)
5. 💰 **Evaluate LeoLabs** (if budget allows)

---

## 🚀 **IMPLEMENTATION PHASES**

### **Phase 1: Immediate (CelesTrak)**
```
✅ No setup required
⚡ Instant 25,000+ objects
🔄 30-second updates
```

### **Phase 2: Enhanced (Space-Track.org)**
```
📝 Register account (24-48 hours)
📊 Access 40,000+ objects
🛰️ Official U.S. data
```

### **Phase 3: Premium (Optional)**
```
💰 LeoLabs for highest accuracy
🔬 ESA for research data
📡 N2YO for real-time tracking
```

---

## ❓ **NEXT STEPS FOR YOU:**

1. **✅ IMMEDIATE:** Test CelesTrak (works right now!)
2. **📝 TODAY:** Register for Space-Track.org account
3. **📱 OPTIONAL:** Sign up for N2YO free tier
4. **🔬 RESEARCH:** Apply for ESA access if needed
5. **💰 EVALUATE:** Consider LeoLabs for premium features

**Let me know which APIs you'd like to set up first, and I'll begin implementing the enhanced data pipeline!** 🛰️

---

**📧 Need Help?** 
- Each API has different approval times
- Some require institutional affiliation
- Happy to help troubleshoot any registration issues! 