import streamlit as st
from langchain.chains import SequentialChain
from langchain import PromptTemplate
from langchain.chains import LLMChain

# ------------------ Streamlit Page Config ------------------

st.markdown(
    """
    <style>
    /* Whole background black */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* Make all text bold */
    html, body, [class*="css"]  {
        font-weight: bold;
        color: #ffffff;
        font-weight:800;
    }

    /* Style the input box */
    .stTextInput>div>div>input {
        background-color: #222222;
        color: #ffffff;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    /* Input hover effect */
    .stTextInput>div>div>input:hover {
        background-color: #333333;
        border: 1px solid #ffffff;
        color: #00ffcc;  /* neon effect on hover */
    }

    /* Style the button */
    .stButton>button {
        background-color: #444444;
        color: #ffffff;
        font-weight: bold;
        border: 1px solid #ffffff;
        transition: all 0.3s ease;
    }

    /* Button hover effect */
    .stButton>button:hover {
        background-color: #00ffcc;
        color: #000000;
        border: 1px solid #00ffcc;
        cursor: pointer;
    }

    /* Style success messages */
    .stSuccess {
        color: #ffffff;
        font-weight: bold;
        background-color: #333333;
        padding: 10px;
        border-radius: 5px;
    }

    /* Style error messages */
    .stError {
        color: #ff4b4b;
        font-weight: bold;
        background-color: #333333;
        padding: 10px;
        border-radius: 5px;
    }

    /* Style the expander */
    .stExpander>div>div {
        background-color: #111111;
        color: #ffffff;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    /* Expander hover effect */
    .stExpander>div>div:hover {
        background-color: #222222;
        color: #00ffcc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('WeMe.ai')

with st.expander("Description"):
    st.write("""
ChatGPT said:

WeMe AI Assistant is an interactive tool that provides real-time, detailed insights about any country, including its capital, population, culture, and more.Using AI capabilities, it can also provide additional context like culture, history, and famous foods, making it an interactive, intelligent, and user-friendly tool for exploring and learning about the world in real time.
""")
import os
from langchain_google_genai import GoogleGenerativeAI  # For Gemini models
# ✅ Set your API key (get it from https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCJ1chuckOp0si1XcAA7t_XSyyzjhaNX_Q"
# ✅ Initialize Gemini LLM
llm = GoogleGenerativeAI( model="gemini-1.5-flash", temperature=0.6)

# 🔹 Custom static data
custom_data = {
    "Deepak": "Deepak आप अब्दुल्ला के दोस्त हैं i am right",
    "Munna": "अख्तर रज़ा आप अब्दुल्ला के दोस्त हैं",
      "Alam": " Khurshid Alam आप अब्दुल्ला के दोस्त हैं",
    "Tashab": "Mohammad Tashab आप अब्दुल्ला के दोस्त हैं",
    "abdullah": "Abdullah is an AI Engineer, YouTuber, Laravel Backend Developer, a future Robot Engineer, and the founder of WeMe.ai.",
    "tahir": "Tahir is Abdullah's brother who supports him in his journey.",
    "habib": "Habib is Abdullah's brother and a Civil Engineer, known for his kind nature and wisdom. He has always supported Abdullah in every step of life, especially during his childhood studies, guiding and motivating him to learn and grow. His encouragement has been a big reason behind Abdullah’s confidence and success.",
    "tayyab": "Tayyab is Abdullah's brother, who supports him in his journey.",
    "jamal": "Jamal is Abdullah's brother and a Mechanical Engineer, energetic and full of new ideas., YouTuber.",
    "fouzan": "Fouzan is Abdullah's nephew, passionate about technology and learning.",
    "ansaree": "Ansaree is Abdullah's mother, a caring and supportive person. She is a source of endless love, sacrifice, and motivation. Abdullah says: 'Amma, your prayers and support are the strength behind my success, and I love you deeply.'",
    "Islam": "Islam is Abdullah's father, always guiding him with wisdom. He is a role model of patience, discipline, and strength. Abdullah says: 'Abba, your guidance is my light, and your teachings inspire me every day. I love you deeply.'",
    "rizwana": "Rizwana is Abdullah’s sister, known for her kind nature.",
    "sultana": "Sultana is Abdullah’s sister, a respected member of the family.",
    "sabbo": "Sabbo is Abdullah’s sister, a loving and cheerful person.",
    "noorsaba": "Noorsaba is Abdullah’s sister, inspiring and supportive.",
    "weme": """1. Country Information:
   - Provides capital cities of any given country.
   - Can include additional context about the country if needed.

2. Famous Foods:
   - Suggests popular or traditional food items from a given country or city.

3. General Knowledge Responses:
   - Answers queries not found in the static knowledge base.
   - Can provide multi-step reasoning (e.g., first find capital → then suggest foods).

4. Multi-Step Generated Responses:
   - Uses SequentialChain to combine multiple LLM tasks dynamically.
   - Produces structured outputs like multiple fields (capital, food) in one query.

5. Flexible AI Responses:
   - Handles open-ended questions beyond the predefined static dataset.
   - Adapts to unexpected queries with intelligent answers generated in real-time.

In summary, the dynamic output features provide real-time, flexible, and structured information across various topics, including countries, foods, and general knowledge. By combining multi-step reasoning and intelligent response generation, the assistant offers a comprehensive and adaptive way for users to access information and explore new topics effectively.""",
"weme.ai": "WeMe.ai is Abdullah's first AI innovation.",
"weme.ai": "WeMe.ai is Abdullah's first AI innovation.",
    "who is the founder" : "Abdullah is the founder of WeMe.ai.",
    "who is your father of weme": "My father is Abdullah, a guiding figure in our family.",
    "who is your trainer": "Abdullah is my trainer and the person who built me."
}



# --------------------------------------------------🔹 Custom static dataend---------------------------------------------------------------------------------
# Create a prompt template
promptt = PromptTemplate(
    input_variables=['country'],   # placeholder (must be provided at runtime)
    template="whats the capital of {country}"   # prompt format
)

chain1 = LLMChain(llm=llm, prompt=promptt, output_key="capital")
prompt_food = PromptTemplate(
    input_variables = ['capital'],
    template = """ suggest some most famous food items of {capital}"""
)

chain2 = LLMChain(llm=llm, prompt=prompt_food, output_key="food")
# Sequential Chain
final_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=['country'],          # only starting input
    output_variables=['capital', 'food']  # final outputs we want
)

user_input = st.text_input("Enter Country")
# if(st.button('Get Food')):
#     response = final_chain.invoke({"country":user_input})
#     st.success(f"Capital: {response['capital']}")
#     st.success(f"Famous Foods: {response['food']}")
if st.button('Get Info'):
    query = user_input.lower().strip()

    if query in custom_data:   # ✅ check static first
        st.success(custom_data[query])
    else:
        try:
            # ✅ Call Gemini chain only if not static
            response = final_chain.invoke({"country": user_input})
            st.success(f"Capital: {response['capital']}")
            st.success(f"Famous Foods: {response['food']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")



