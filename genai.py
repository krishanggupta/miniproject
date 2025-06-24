import google.generativeai as genai

# ✅ Directly configure with your actual API key (do NOT use this in production)
genai.configure(api_key="AIzaSyAIhrSVwoKg69NrmWhE_e-B34zkWz5nZJ8")

# ✅ Initialize the model
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# ✅ Generate content
response = model.generate_content("Write a short report about Diabetic Retinopathy for a patient.")

# ✅ Print the response
print(response.text)
