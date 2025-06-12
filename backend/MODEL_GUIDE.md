# AI Model Configuration Guide

## 🤖 How to Change the Default AI Model

### **Method 1: Environment Variable (Recommended)**

Add to your `.env` file:
```bash
DEFAULT_AI_MODEL=gemini-2.0-flash-001
```

### **Method 2: Direct API Request**

Specify model in each request:
```json
{
  "json_data": "...",
  "model": "gemini-2.0-flash-001",
  "processing_mode": "ai_only"
}
```

## 📋 **Available Gemini Models**

### **Gemini 2.0 Models (Latest)**
- `gemini-2.0-flash-001` ⭐ **RECOMMENDED** - Latest, fastest
- `gemini-2.0-flash-experimental` - Cutting edge features

### **Gemini 1.5 Models (Stable)**
- `gemini-1.5-flash-001` - Fast, reliable
- `gemini-1.5-flash-002` - Improved version
- `gemini-1.5-pro-001` - Higher quality, slower
- `gemini-1.5-pro-002` - Latest pro version

### **Legacy Models**
- `gemini-2.5-flash-preview-05-20` - Previous default
- `gemini-1.0-pro` - Original Gemini

## 🔧 **Configuration Priority**

1. **API Request** - Model specified in request body
2. **Environment Variable** - `DEFAULT_AI_MODEL` in .env
3. **Code Default** - Fallback in config.py

## 🧪 **Testing Different Models**

```bash
# Test with specific model
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": "{\"revenue\": 1000000, \"currency\": \"EUR\"}",
    "model": "gemini-2.0-flash-001",
    "processing_mode": "ai_only"
  }'
```

## 📊 **Model Comparison**

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| `gemini-2.0-flash-001` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | **Production** |
| `gemini-1.5-pro-002` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰 | Complex analysis |
| `gemini-1.5-flash-002` | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | Fast processing |

## 🚀 **Quick Setup**

1. **Update .env file:**
   ```bash
   cp .env.example .env
   # Edit .env and set DEFAULT_AI_MODEL=gemini-2.0-flash-001
   ```

2. **Restart server:**
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Test new model:**
   ```bash
   python test_ai.py
   ```

## 💡 **Tips**

- **Use Gemini 2.0** for best performance/cost ratio
- **Use Gemini 1.5 Pro** for complex financial analysis
- **Test different models** with your specific data
- **Monitor costs** in Google Cloud Console

## 🔍 **Model Capabilities**

All models support:
- ✅ Multi-currency conversion
- ✅ Financial analysis
- ✅ Excel generation
- ✅ Chart creation
- ✅ Executive summaries
- ✅ Session memory

**Gemini 2.0** additions:
- ⚡ 2x faster processing
- 🧠 Better reasoning
- 💰 Lower costs
- 🔄 Improved streaming