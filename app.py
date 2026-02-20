import streamlit as st
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import  ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal, TypedDict
import pandas as pd

# --- setup ---
load_dotenv(Path(__file__).parent / ".env")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

# --- pydantic schema for the output ---
class Ingredient(BaseModel):
    name_en: str
    name_nl: str
    quantity: Optional[str] = "to taste"

class RecipeStep(BaseModel):
    display_name_en: str = Field(description="Name of ingredient or tool, e.g. 'Onion'")
    display_name_nl: str = Field(description="Same in Dutch, e.g. 'Ui'")
    action_en: str = Field(description="Short verb phrase max 3 words, e.g. 'Wash and chop'")
    action_nl: str = Field(description="Same in Dutch, e.g. 'Wassen en snijden'")
    instructions_en: str = Field(description="Full instruction, clear for a 12-year-old")
    instructions_nl: str = Field(description="Same in Dutch")
    workplace: Literal["Stove", "Oven", "Blender", "Cutting board", "Bowl", "Pan"]
    has_timer: bool
    timer_minutes: Optional[float] = None
    is_first_appearance: bool

    # llm keeps returning wrong workplace values so we map them here
    @field_validator("workplace", mode="before")
    @classmethod
    def fix_workplace(cls, v):
        mapping = {
            "Plate": "Bowl",
            "Wok": "Stove",
            "Pot": "Stove",
            "Stock pot": "Stove",
            "Stock Pot": "Stove",
            "Sink": "Bowl",
            "Counter": "Cutting board",
            "Table": "Cutting board",
            "Grill": "Stove",
            "Microwave": "Oven",
            "Chopping board": "Cutting board",
            "Chopping Board": "Cutting board",
            "Toaster": "Stove",
            "Worktop": "Cutting board",
            "Kitchen counter": "Cutting board",
            "Loaf pan": "Oven",
            "Loaf Pan": "Oven",
        }
        return mapping.get(v, v)

class SousChefRecipe(BaseModel):
    generic_name_en: str
    generic_name_nl: str
    recipe_name_en: str
    recipe_name_nl: str
    description_en: str
    description_nl: str
    difficulty: Literal["Easy", "Medium", "Intermediate"]
    servings: int
    prep_time_minutes: int
    cook_time_minutes: int
    calories_per_person: Optional[int] = 0
    protein_g: Optional[float] = 0
    carbs_g: Optional[float] = 0
    fat_g: Optional[float] = 0
    ingredients: List[Ingredient]
    steps: List[RecipeStep]

    @model_validator(mode="before")
    @classmethod
    def fix_field_names(cls, v):
        renames = {
            "calories_per_serving": "calories_per_person",
            "protein_g_per_serving": "protein_g",
            "carbs_g_per_serving": "carbs_g",
            "fat_g_per_serving": "fat_g",
        }
        for old, new in renames.items():
            if old in v and new not in v:
                v[new] = v.pop(old)
        return v

    # same problem as workplace - llm returns "Moderate", "Hard" etc
    @field_validator("difficulty", mode="before")
    @classmethod
    def fix_difficulty(cls, v):
        mapping = {
            "Moderate": "Medium",
            "Hard": "Intermediate",
            "Advanced": "Intermediate",
            "Beginner": "Easy",
            "Simple": "Easy"
        }
        return mapping.get(v, v)

    @model_validator(mode="after")
    def fix_difficulty_by_steps(self):
        # override llm difficulty with rule based on step count
        # llm is inconsistent (scrambled eggs got Medium, greek salad got Intermediate)
        step_count = len(self.steps)
        if step_count <= 6:
            self.difficulty = "Easy"
        elif step_count <= 11:
            self.difficulty = "Medium"
        else:
            self.difficulty = "Intermediate"
        return self

# --- agent state ---
class AgentState(TypedDict, total=False):
    raw_recipe: str
    parsed_recipe: SousChefRecipe
    critique: str
    final_recipe: SousChefRecipe
    attempts: int

# --- prompts for the agent ---
transform_prompt = """
### Role
You are a professional recipe editor for SousChef, a cooking app for beginners.

### Task
Transform the raw recipe into SousChef format. You MUST use EXACTLY these JSON field names.

### Required JSON structure:
{{
  "generic_name_en": "Chicken Rice",
  "generic_name_nl": "Kip Rijst",
  "recipe_name_en": "One-pan Chicken and Garlic Rice",
  "recipe_name_nl": "Kip met knoflookrijst in één pan",
  "description_en": "Paragraph 1 about ease and taste. Paragraph 2 about SousChef guidance.",
  "description_nl": "Paragraaf 1. Paragraaf 2.",
  "difficulty": "Easy",
  "servings": 4,
  "prep_time_minutes": 10,
  "cook_time_minutes": 25,
  "calories_per_person": 550,
  "protein_g": 35,
  "carbs_g": 45,
  "fat_g": 18,
  "ingredients": [
    {{"name_en": "Onion", "name_nl": "Ui", "quantity": "1 piece"}}
  ],
  "steps": [
    {{
      "display_name_en": "Onion",
      "display_name_nl": "Ui",
      "action_en": "Peel and chop",
      "action_nl": "Schillen en snijden",
      "instructions_en": "Peel the onion and chop it into small pieces.",
      "instructions_nl": "Schil de ui en snijd hem in kleine stukjes.",
      "workplace": "Cutting board",
      "has_timer": false,
      "timer_minutes": null,
      "is_first_appearance": true
    }}
  ]
}}

### Rules
- workplace must be ONLY one of: "Stove", "Oven", "Blender", "Cutting board", "Bowl", "Pan"
- difficulty must be ONLY one of: "Easy", "Medium", "Intermediate"
- One micro-step per action, instructions clear for a 12-year-old
- is_first_appearance: true only on first use of that ingredient
- generic_name_en: short generic name (e.g. "Chicken Rice" not the full recipe name)

### Raw Recipe
{raw_recipe}

### Output
Return ONLY valid JSON. Nothing else.
"""

critique_prompt = """
### Role
You are a strict quality checker for SousChef recipes.

### Check ALL of the following:
1. Are instructions clear enough for a 12-year-old?
2. Are timers set for ALL steps that involve waiting or cooking time?
3. Is difficulty appropriate? (Easy=few steps, Medium=some complexity, Intermediate=complex)
4. Are Dutch translations present and correct for ALL text fields?
5. Is is_first_appearance=true only on the FIRST use of each ingredient?
6. Are workplace values only from: Stove, Oven, Blender, Cutting board, Bowl, Pan?

### Recipe to review:
{recipe_json}

### Output
Be specific about each problem. If everything is correct, write ONLY: APPROVED
"""

repair_prompt = """
### Role
You are a SousChef recipe editor. Fix ALL problems listed in the critique.

### Original Recipe:
{recipe_json}

### Problems to fix:
{critique}

### Rules (do not violate these):
- workplace must be ONLY: "Stove", "Oven", "Blender", "Cutting board", "Bowl", "Pan"
- difficulty must be ONLY: "Easy", "Medium", "Intermediate"

### Output
Return ONLY the corrected JSON. Nothing else.
"""

# --- agent nodes ---

def transform_node(state):
    """takes raw recipe text and converts to souschef json"""
    structured_llm = llm.with_structured_output(SousChefRecipe, method="json_mode")
    prompt = transform_prompt.format(raw_recipe=state["raw_recipe"])
    recipe = structured_llm.invoke(prompt)
    return {"parsed_recipe": recipe, "attempts": 0}

def critique_node(state):
    """checks the recipe for problems"""
    recipe_json = state["parsed_recipe"].model_dump_json(indent=2)
    prompt = critique_prompt.format(recipe_json=recipe_json)
    response = llm.invoke(prompt)
    return {"critique": response.content}

def repair_node(state):
    """fixes problems found by critique"""
    structured_llm = llm.with_structured_output(SousChefRecipe, method="json_mode")
    recipe_json = state["parsed_recipe"].model_dump_json(indent=2)
    prompt = repair_prompt.format(recipe_json=recipe_json, critique=state["critique"])
    fixed = structured_llm.invoke(prompt)
    # write back to parsed_recipe so critique can check it again
    return {"parsed_recipe": fixed, "attempts": state.get("attempts", 0) + 1}

def finalize_node(state):
    """just saves the final version"""
    return {"final_recipe": state["parsed_recipe"]}

def should_repair_or_finalize(state):
    """routing logic - approved goes to finalize, otherwise repair (max 2 times)"""
    critique = state.get("critique", "")
    attempts = state.get("attempts", 0)
    if "APPROVED" in critique:
        return "finalize"
    elif attempts >= 2:
        return "finalize"
    else:
        return "repair"

# --- build the graph ---
# the key thing is repair -> critique edge, this makes the loop
graph = StateGraph(AgentState)
graph.add_node("transform", transform_node)
graph.add_node("critique", critique_node)
graph.add_node("repair", repair_node)
graph.add_node("finalize", finalize_node)
graph.add_edge(START, "transform")
graph.add_edge("transform", "critique")
graph.add_conditional_edges("critique", should_repair_or_finalize, {
    "repair": "repair",
    "finalize": "finalize"
})
graph.add_edge("repair", "critique")  # this is the loop - repair goes back to critique
graph.add_edge("finalize", END)
agent = graph.compile()

# --- streamlit ui ---
st.set_page_config(page_title="SousChef recipe transformer", page_icon="🍳", layout="wide")

st.title("SousChef recipe transformer")
st.markdown("Transform any recipe into SousChef step-by-step format using AI")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Recipe")
    recipe_input = st.text_area(
        "Paste your recipe here:",
        height=400,
        placeholder="Paste any recipe text here...\n\nFor example:\nSpaghetti Carbonara\nServings: 4\n\nIngredients:\n200g pancetta\n4 eggs\n..."
    )
    transform_btn = st.button("Transform recipe", type="primary", use_container_width=True)

with col2:
    st.subheader("SousChef output")

    if transform_btn and recipe_input:
        with st.spinner("Transforming recipe..."):
            try:
                result = agent.invoke({"raw_recipe": recipe_input})
                recipe = result["final_recipe"]
                st.session_state["last_recipe"] = recipe

                st.success(f"Done: {recipe.recipe_name_en}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Difficulty", recipe.difficulty)
                m2.metric("Servings", recipe.servings)
                m3.metric("Prep", f"{recipe.prep_time_minutes} min")
                m4.metric("Cook", f"{recipe.cook_time_minutes} min")

                st.markdown("**Nutrition per person**")
                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Calories", f"{recipe.calories_per_person} kcal")
                n2.metric("Protein", f"{recipe.protein_g}g")
                n3.metric("Carbs", f"{recipe.carbs_g}g")
                n4.metric("Fat", f"{recipe.fat_g}g")

                st.markdown("**Description**")
                st.info(recipe.description_en)

                st.markdown(f"**Steps ({len(recipe.steps)} total)**")
                for i, step in enumerate(recipe.steps, 1):
                    timer = f" | {int(step.timer_minutes)} min" if step.has_timer and step.timer_minutes else ""
                    with st.expander(f"Step {i}: {step.display_name_en} - {step.action_en}{timer}"):
                        st.write(f"**{step.workplace}**")
                        st.write(step.instructions_en)
                        st.caption(f"NL: {step.instructions_nl}")

                st.markdown("**Export**")
                e1, e2 = st.columns(2)

                json_data = json.dumps(recipe.model_dump(), indent=2, ensure_ascii=False)
                e1.download_button("Download JSON", json_data,
                                   file_name=f"{recipe.generic_name_en.lower().replace(' ', '_')}.json",
                                   mime="application/json", use_container_width=True)

                steps_df = pd.DataFrame([{
                    "step": i+1,
                    "display_name_en": s.display_name_en,
                    "action_en": s.action_en,
                    "instructions_en": s.instructions_en,
                    "workplace": s.workplace,
                    "has_timer": s.has_timer,
                    "timer_minutes": s.timer_minutes
                } for i, s in enumerate(recipe.steps)])

                e2.download_button("Download CSV", steps_df.to_csv(index=False),
                                   file_name=f"{recipe.generic_name_en.lower().replace(' ', '_')}.csv",
                                   mime="text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif transform_btn and not recipe_input:
        st.warning("Please paste a recipe first!")
    else:
        st.info("Paste a recipe on the left and click Transform")