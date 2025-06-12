# 🐛 Debug Mode Guide

## 🎯 **Clean Debug Mode - See AI Thinking Without Your Data!**

### **✅ What You Get Now:**
- 🤖 **AI Reasoning Steps** - See how the AI thinks through problems
- 🔧 **Tool Calls** - Watch AI use GoogleSearch, PythonTools, etc.
- 📊 **Response Analysis** - Content length and preview
- 💭 **Step-by-step Logic** - First 3 reasoning steps shown
- 🛠️  **Tool Usage** - Which tools were executed

### **❌ What You DON'T Get:**
- ❌ Full JSON input data echoed back
- ❌ Massive log dumps
- ❌ Sensitive data exposure
- ❌ Cluttered output

## 🔧 **How to Enable Clean Debug Mode**

### **Method 1: Environment Variable**
```bash
# Add to your .env file:
AGNO_DEBUG=true
```

### **Method 2: Temporary Testing**
```bash
# Run server with debug enabled:
AGNO_DEBUG=true uvicorn app.main:app --reload
```

## 📋 **Example Debug Output**

```
🤖 AI PROMPT SUMMARY:
   📁 Working Directory: /tmp/request_abc123
   📄 File Name: financial_report
   📝 Description: Q4 2024 earnings data
   📊 JSON Data Size: 15,847 characters
   🔍 JSON Preview: {"company": "TechCorp", "revenue": 5000000...
   🎯 Task: Financial Excel report with currency conversion

🧠 AI RESPONSE ANALYSIS:
   🤔 Reasoning Steps: 8 steps captured
   💭 Step 1: I need to analyze this financial data and create a comprehensive Excel report...
   💭 Step 2: First, I'll examine the currency information to determine conversion needs...
   💭 Step 3: I'll structure the report with multiple sheets for different data categories...
   🔧 Tool Calls: 3 executed
   🛠️  Tool 1: GoogleSearchTools
   🛠️  Tool 2: PythonTools  
   🛠️  Tool 3: PythonTools
   📄 Response Length: 2,847 characters
   📋 Response Preview: # Financial Analysis Report\n\n## Process Overview\nI've successfully...
```

## 🚀 **Perfect for Debugging:**

### **✅ When AI Reasoning Fails:**
See exactly where the AI's logic breaks down

### **✅ When Tools Don't Work:**
Check which tools were called and if they succeeded

### **✅ When Currency Conversion Issues:**
See if GoogleSearch tool found exchange rates

### **✅ When Excel Generation Problems:**
Watch PythonTools execute the script step-by-step

### **✅ When Performance Issues:**
Monitor reasoning complexity and tool usage

## 💡 **Debug Tips:**

1. **Enable debug mode** for development/testing
2. **Disable debug mode** for production (cleaner logs)
3. **Check reasoning steps** if AI makes poor decisions
4. **Monitor tool calls** to ensure proper execution
5. **Watch JSON preview** to verify data parsing

## 🔄 **Toggle Debug Mode Live:**

```bash
# Enable debug
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"json_data": "...", "processing_mode": "ai_only"}'

# Check logs for:
# 🤖 AI PROMPT SUMMARY
# 🧠 AI RESPONSE ANALYSIS
# 🤔 Reasoning Steps
# 🔧 Tool Calls
```

## 🎯 **What This Solves:**

- ✅ **Privacy**: No sensitive JSON data in logs
- ✅ **Clarity**: Clean, structured debug output
- ✅ **Insight**: See AI decision-making process
- ✅ **Debugging**: Track down issues quickly
- ✅ **Performance**: Monitor AI efficiency

Now you can see **exactly how the AI thinks** without your JSON data cluttering up the output! 🧠✨