# SousChef recipe transformer

AI agent that takes normal recipe text and converts it into SousChef micro-step format. Built with LangGraph.

## Project structure
```
souschef/
├── souschef_agent.ipynb  # main notebook - agent + evaluation
├── app.py                # streamlit web app
├── recipes/              # output files (10 recipes in json/csv)
└── .env                  # api key (not in repo)
```

## How it works

The agent is basically a LangGraph workflow with 3 main nodes and a loop:

1. **Transform** - LLM takes raw recipe text and converts it to SousChef JSON format
2. **Critique** - another LLM call checks the result for problems (wrong workplace values, missing translations, etc)
3. **Repair** - if critique found issues, LLM fixes them and sends back to Critique again
4. **Finalize** - when critique says APPROVED or we already tried 2 repairs, save the result

The important thing here is that Repair goes back to Critique, not straight to Finalize. So it is actually a loop, not just one pass. I think this is what makes it different from a simple chain.

## Evaluation

I tested on 10 different recipes: scrambled eggs, banana bread, tomato soup, pasta bolognese, vegetable curry, chocolate cake, greek salad, cheese omelette, lentil soup, pancakes.

All 10 passed Pydantic validation.

| Metric | Result |
|--------|--------|
| Validation pass rate | 10/10 |
| Passed on first critique | 6/10 |
| Needed repair cycle | 4/10 |
| Average steps per recipe | around 10 |

## Why I made certain decisions

**Why LangGraph?**
I needed conditional edges to make the critique/repair loop work. With a simple LangChain chain you cant route back from repair to critique. LangGraph lets you do this with conditional_edges, which was exactly what I needed.

**Why llama-3.1-8b through Groq?**
Mainly because its free and fast. For this assignment its enough to get valid JSON output with Pydantic. Of course a bigger model like llama-3.1-70b or Claude would give better Dutch translations and more accurate difficulty levels, but for a prototype this works.

**Why temperature=0.3?**
I tried different values. With 0 or 0.1 the output was too rigid and ignored some context from the recipe. With 0.5+ the field names started to be inconsistent (sometimes "workplace" sometimes "work_place" etc). 0.3 seemed like a good balance.

**Why Pydantic validators?**
The LLM keeps inventing new workplace values even when I explicitly tell it to only use the 6 allowed ones. It would return "Chopping board", "Wok", "Stock pot", "Toaster" and so on. Same with difficulty - "Moderate" instead of "Medium", "Beginner" instead of "Easy". So I added field_validator to automatically map these wrong values to correct ones. I also added a model_validator that catches when the LLM returns wrong field names like "calories_per_serving" instead of "calories_per_person", so it doesnt break validation.

**Why rule-based difficulty?**
The LLM was very bad at setting difficulty - scrambled eggs got Medium, greek salad got Intermediate, pasta bolognese got Easy. The critique node didnt catch this either. So I just made a simple rule based on step count: 6 or less steps = Easy, 7 to 11 = Medium, 12+ = Intermediate. Its not perfect but way more consistent than what the LLM was giving.

## Problems and limitations

- **Dutch translations** are not great. The model translates word by word instead of using natural Dutch. For example it wrote "Gesmolten eieren" (which means melted eggs) instead of "Roerei" for scrambled eggs. This is probably the biggest quality issue.
- **Difficulty was inconsistent** - the LLM gave wrong difficulty levels all the time (scrambled eggs = Medium, greek salad = Intermediate). Fixed it with a rule that sets difficulty by step count, works much better than trusting the LLM for this.
- **Workplace values** - as I mentioned, the LLM keeps inveting new ones even with strict instructions. I solved it with validators but it shows that LLMs are not reliable with enums.
- **Schema is simplified** - the real SousChef CMS would need quantity split into amount (number) and metric_unit (enum), duration in seconds for each step, and more nutrition fields. I kept it simpler for the prototype.

## What I would improve

1. Use a stronger model (70b or Claude) - this would fix most Dutch translation problems
2. Parse quantities properly - split "2 tbsp" into amount=2 and unit="tbsp" to match the real CMS
3. Add some kind of quality score - like 0-10 for each recipe based on completeness and translation quality
4. Compare against manually written SousChef recipes to measure how close the AI gets
5. Add timer reminder text ("Tick off this step, then I'll start the timer for you") - I tried this but it caused the repair loop to go infinitely with the small model so I removed it

## How to run

```bash
source venv/bin/activate
streamlit run app.py
```

## Tech stack

- LangGraph - agent workflow with conditional edges
- LangChain + Groq (llama-3.1-8b-instant) - LLM calls
- Pydantic v2 - schema validation with field validators
- Streamlit - web interface with JSON/CSV export
- python-dotenv - for api key# souschef-recipe-transformer
