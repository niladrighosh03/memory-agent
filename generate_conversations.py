import os
import json
import random
import csv
import re
import time
import math
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5-mini"

# --- 1. Utils ---

def gen_utterance(prompt_: str, system_prompt: str = "You are a helpful assistant.", 
                  temperature: float = 1.0, max_len: int = 1000) -> str:
    """
    Generates text using gpt-5-mini.
    NOTE: temperature must be 1.0 for this model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # temperature=1.0, # gpt-5-mini only supports default 1.0
            max_completion_tokens=max_len 
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT Generation Error: {e}")
        return ""

# --- 2. Persona Definitions ---

class Driver(BaseModel):
    role: str
    age: int

class DrivingUsage(BaseModel):
    commute_distance_km: str
    driving_environment: str
    daily_parking: str
    business_use: bool
    annual_mileage_band: str
    primary_drivers: List[Driver]

class SegmentProfile(BaseModel):
    price_sensitivity: str
    buyer_type: str
    switching_reason: str
    claims_trauma: str

class ServicePreferences(BaseModel):
    communication_style: str
    decision_style: str
    information_format: str
    preferred_channel: str

class ProductPreferences(BaseModel):
    coverage_goal: str
    deductible_appetite: str
    valued_addons: List[str]

class Persona(BaseModel):
    persona_id: str
    archetype: str
    driving_usage_profile: DrivingUsage
    customer_segment_profile: SegmentProfile
    service_preferences: ServicePreferences
    product_preferences: ProductPreferences

class PersonaList(BaseModel):
    items: List[Persona]

ARCHETYPES = [
    "Young urban professional",
    "Student / early career youth",
    # "Family household with kids",
    # "Self-employed / small business owner",
    # "Gig worker / delivery driver",
    # "Corporate executive / high income owner",
    # "Homemaker / occasional driver",
    # "Senior citizen / retired",
    # "Rural or semi-urban resident",
    "Budget-conscious daily commuter"
]

def generate_personas(n_per_archetype=1, archetypes_list=None) -> List[Dict]:
    all_personas_dict = []
    pid = 1
    
    target_archetypes = archetypes_list if archetypes_list else ARCHETYPES
    
    print(f"Generating {n_per_archetype} personas per archetype for {len(target_archetypes)} archetypes...")
    for arch in tqdm(target_archetypes):
        prompt = f"Generate exactly {n_per_archetype} realistic Indian motor insurance customer personas for archetype: {arch}. Fill all fields realistically and vary behavior."
        
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a synthetic data generator."}, 
                    {"role": "user", "content": prompt}
                ],
                response_format=PersonaList,
            )
            
            personas = completion.choices[0].message.parsed.items
            
            for p in personas:
                p.persona_id = f"P_{pid:03d}"
                p.archetype = arch
                all_personas_dict.append(p.model_dump())
                pid += 1
                
        except Exception as e:
            print(f"Error generating {arch}: {e}")
            
    return all_personas_dict

# --- 3. Expert Agents ---

class ExpertAgents:
    """Multi-agent reasoning system for persuasion"""
    
    def __init__(self):
        self.experts = {
            "Engagement Expert": self._engagement_reasoning,
            "Keyterm Expert": self._keyterm_reasoning,
            "Intent Expert": self._intent_reasoning,
            "Sentiment Expert": self._sentiment_reasoning,
            "Knowledge Expert": self._knowledge_reasoning
        }
    
    def _engagement_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        prompt = f"""Analyze briefly in ONE short sentence.
User: "{user_message}"
Turn: {len(history)}
Provide ONE brief sentence about engagement strategy (tone, rapport, timing)."""
        return gen_utterance(prompt, system_prompt="You are an Engagement Expert.", max_len=150)
    
    def _keyterm_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        prompt = f"""Identify key points briefly.
User: "{user_message}"
Vehicle: {context.get('vehicle_value_category', 'unknown')}
ONE brief sentence identifying key features to emphasize."""
        return gen_utterance(prompt, system_prompt="You are a Keyterm Expert.", max_len=150)
    
    def _intent_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        prompt = f"""Classify intent briefly.
User: "{user_message}"
Classify as: Request quote, Ask coverage, Express concern, Request info, Negotiate price, or Confirm/Reject.
ONE brief sentence only."""
        return gen_utterance(prompt, system_prompt="You are an Intent Expert.", max_len=150)
    
    def _sentiment_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        prompt = f"""Analyze tone briefly.
User: "{user_message}"
ONE brief sentence about emotional state and response approach."""
        return gen_utterance(prompt, system_prompt="You are a Sentiment Expert.", max_len=150)
    
    def _knowledge_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        prompt = f"""Identify needed information briefly.
User: "{user_message}"
Provider: {context.get('recommended_provider', 'unknown')}
ONE brief sentence about what knowledge to use."""
        return gen_utterance(prompt, system_prompt="You are a Knowledge Expert.", max_len=150)
    
    def orchestrate(self, context: Dict, user_message: str, history: List) -> Dict[str, str]:
        # Dynamically select experts
        active_experts = ["Engagement Expert"] # Always active
        msg_lower = user_message.lower()
        
        if len(history) <= 2:
            active_experts.extend(["Keyterm Expert", "Knowledge Expert"])
        
        active_experts.append("Intent Expert")
        
        if any(word in msg_lower for word in ["concern", "worried", "sure", "expensive", "discount"]):
            active_experts.append("Sentiment Expert")
        if any(word in msg_lower for word in ["feature", "cover", "benefit"]):
            active_experts.append("Keyterm Expert")
        if any(word in msg_lower for word in ["price", "cost", "detail", "info"]):
            active_experts.append("Knowledge Expert")
            
        active_experts = list(set(active_experts))[:4] # Limit to 4
        
        thinking = {}
        for expert_name in active_experts:
            try:
                reasoning = self.experts[expert_name](context, user_message, history)
                thinking[expert_name] = reasoning
            except Exception as e:
                thinking[expert_name] = f"Error: {e}"
        return thinking
    
    def synthesize(self, thinking: Dict[str, str], context: Dict) -> str:
        experts_summary = "\n".join([f"{name}: {thought}" for name, thought in thinking.items()])
        prompt = f"""Synthesize expert insights into ONE brief sentence.
Expert Analysis:
{experts_summary}
ONE sentence overall strategy."""
        return gen_utterance(prompt, system_prompt="You are a Strategy Synthesizer.", max_len=150)

# --- 4. Dialogue Generator ---

class DialogueGenerator:
    def __init__(self, insurance_data):
        self.insurance_data = insurance_data
        self.conversation_context = {}
        self.conversation_history = []
        self.intent_history = []
        self.strategy_history = []
        self.thinking_log = []
        self.experts = ExpertAgents()
        
        self.insurers = {name.lower(): name for name in self.insurance_data["Motor Insurance"]["description"].keys()}
        
        self.agent_strategies = [
            "Default", "Credibility", "Emotional", "Logical", "Personal", "Persona"
        ]
        
        # Transitions
        self.intent_transitions = {
            "Initial": ["Ask Coverage Details", "Request quote", "Request additional info"],
            "Request quote": ["Negotiate price", "Ask Coverage Details", "Express Concern"],
            "Ask Coverage Details": ["Express Concern", "Request quote", "Request additional info"],
            "Express Concern": ["Request additional info", "Negotiate price", "Ask Coverage Details"],
            "Request additional info": ["Request quote", "Express Concern", "Negotiate price"],
            "Negotiate price": ["Confirm Plan", "Express Concern", "Reject Offer"]
        }
        
        self.deductible_ranges = {
            "Economy": {"min": 1000, "standard": 2500, "max": 5000},
            "Mid-range": {"min": 2500, "standard": 5000, "max": 10000},
            "Premium": {"min": 5000, "standard": 10000, "max": 25000}
        }

    def _extract_price(self, text: str) -> Optional[int]:
        match = re.search(r'₹\s*([\d,]+)|([\d,]+)\s*rs', text, re.IGNORECASE)
        if match:
            price_str = (match.group(1) or match.group(2)).replace(',', '')
            return int(price_str)
        return None

    def _retrieve_relevant_info(self, query: str, recommended_provider: str = None) -> Dict:
        # Simplified retrieval logic
        data = self.insurance_data["Motor Insurance"]
        return {
            "Description": data["description"].get(recommended_provider, "Standard Provider"),
            "Features": data["insurer_specific_features"].get(recommended_provider, []),
            "Pricing": data["pricing_information"]["comprehensive_premium_ranges"].get(recommended_provider, {})
        }

    def _get_next_intent(self, last_intent: str, persona: Dict, context: Dict) -> str:
        turn_count = context.get('turn_count', 0)
        
        # Enforce 22-28 turns range
        # Block finalization before turn 20
        if turn_count < 20:
            possible = self.intent_transitions.get(last_intent, ["Request additional info"])
            possible = [i for i in possible if i not in ["Confirm Plan", "Reject Offer"]]
            if not possible: possible = ["Ask Coverage Details"]
            return random.choice(possible)
            
        # Force finalization after turn 22
        if turn_count >= 22:
            if not context.get('premium_offered'):
                 return "Request quote"
            if context['outcome'] == 'Accept':
                return "Confirm Plan"
            else:
                return "Reject Offer"
                
        # Between 20-22, can transition normally
        possible = self.intent_transitions.get(last_intent, ["Request additional info"])
        return random.choice(possible)

    def generate_agent_response(self, user_message: str, context: Dict, conversation_history: List, 
                               conversation_id: int, turn_no: int,
                               recommended_provider: str = None) -> Tuple[str, str]:
        
        # 1. Orchestrate Experts
        expert_thinking = self.experts.orchestrate(context, user_message, conversation_history)
        
        # Log thinking
        for name, thought in expert_thinking.items():
            self.thinking_log.append({
                "conversation_id": conversation_id, "turn_no": turn_no, 
                "agent_name": name, "thinking": thought
            })
            
        # Synthesize
        orchestrator_summary = self.experts.synthesize(expert_thinking, context)
        self.thinking_log.append({
            "conversation_id": conversation_id, "turn_no": turn_no, 
            "agent_name": "Orchestrator", "thinking": orchestrator_summary
        })
        
        # Strategy Selection
        chosen_strategy = random.choice(self.agent_strategies)
        
        # Context & Instructions
        vehicle_category = context.get('vehicle_value_category', 'Mid-range')
        provider_info = self._retrieve_relevant_info(user_message, recommended_provider)
        
        action_instruction = "Respond naturally."
        if turn_no == 2: # First agent response (turn 1 is user)
            action_instruction = f"Acknowledge the new {vehicle_category} car. Recommend {recommended_provider}."
        elif "price" in user_message.lower() and not context.get('premium_offered'):
            base = {'Economy': 15000, 'Mid-range': 25000, 'Premium': 40000}.get(vehicle_category, 25000)
            price = int(base * random.uniform(1.05, 1.25))
            self.conversation_context['premium_offered'] = price
            action_instruction = f"Quote ₹{price:,}. Explain value."
        elif "negotiate" in user_message.lower() or "expensive" in user_message.lower():
             if context.get('negotiation_stage', 0) == 0:
                 self.conversation_context['negotiation_stage'] = 1
                 action_instruction = "Justify price. Do NOT discount yet."
             else:
                 price = int(context['premium_offered'] * 0.92)
                 self.conversation_context['premium_offered'] = price
                 self.conversation_context['negotiation_stage'] += 1
                 action_instruction = f"Offer final price ₹{price:,}."

        history_str = "\n".join([f"{speaker}: {msg}" for _, _, speaker, msg in conversation_history[-6:]])
        
        prompt = f"""
You are an insurance agent. Strategy: {chosen_strategy}
Expert Insights: {orchestrator_summary}
History:
{history_str}
User Content Analysis: {context.get('vehicle_info', '')}
Goal: {action_instruction}
Provider Info: {json.dumps(provider_info)}
CRITICAL: Use Indian Rupees (₹) for money. Be conversational.
"""
        response = gen_utterance(prompt, system_prompt="You are a helpful insurance agent.", max_len=600)
        
        # Extract price if agent mentioned it
        price = self._extract_price(response)
        if price: self.conversation_context['premium_offered'] = price
        
        return chosen_strategy, response

    def generate_user_utterance(self, agent_message: str, context: Dict, conversation_history: List, 
                               persona: Dict) -> Tuple[str, str]:
        
        history_str = "\n".join([f"{speaker}: {msg}" for _, _, speaker, msg in conversation_history[-6:]])
        
        last_intent = self.intent_history[-1] if self.intent_history else "Initial"
        chosen_intent = self._get_next_intent(last_intent, persona, context)
        self.intent_history.append(chosen_intent)
        
        # Construct rich persona string
        persona_desc = f"""
Archetype: {persona['archetype']}
Role: {persona['driving_usage_profile']['primary_drivers'][0]['role']}, Age: {persona['driving_usage_profile']['primary_drivers'][0]['age']}
Price Sensitivity: {persona['customer_segment_profile']['price_sensitivity']}
Communication Style: {persona['service_preferences']['communication_style']}
Deductible Preference: {persona['product_preferences']['deductible_appetite']}
"""
        
        instruction = f"Respond with intent: {chosen_intent}."
        if chosen_intent == "Confirm Plan":
            instruction += f" Accept the offer of ₹{context.get('premium_offered', 'N/A')}."
        elif chosen_intent == "Reject Offer":
            instruction += f" Reject the offer. It is above your budget."
        elif chosen_intent == "Negotiate price":
            instruction += " Ask for a better deal or discount."

        prompt = f"""
You are a customer interacting with an insurance agent.
Your Persona:
{persona_desc}

Conversation:
{history_str}

Task: {instruction}
Be realistic, keep it to 1-2 sentences. Use Indian Rupees (₹) if discussing money.
"""
        response = gen_utterance(prompt, system_prompt="You are a customized user persona.", max_len=400)
        return chosen_intent, response

    def analyze_vehicle(self) -> Dict:
        """
        Simulates vehicle analysis since no image is provided.
        Generates a random vehicle profile.
        """
        # Randomly select a category
        category = random.choice(["Economy", "Mid-range", "Premium"])
        
        # Generate a description based on category
        prompt = f"""Generate a brief visual description of a {category} car in India. 
        Include color, body type (SUV/Sedan/Hatchback), and condition. 
        Example: 'A shiny red Maruti Swift hatchback in excellent condition.'"""
        
        description = gen_utterance(prompt, max_len=300)
        
        return {"vehicle_category": category, "description": description}

    def generate_conversation(self, conversation_id: int, persona: Dict) -> Tuple[List, List]:
        # Initialize for new conversation
        self.conversation_history = []
        self.intent_history = []
        self.thinking_log = []
        conversation_export = []
        
        # 1. Analyze Vehicle (Simulated)
        vehicle_analysis = self.analyze_vehicle()
        
        # 2. Setup Context
        # Decide outcome based on persona price sensitivity (roughly)
        sensitivity = persona['customer_segment_profile']['price_sensitivity'].lower()
        if "high" in sensitivity: 
            outcome = "Reject" if random.random() < 0.6 else "Accept"
        else:
            outcome = "Accept" if random.random() < 0.8 else "Reject"
            
        provider = random.choice(list(self.insurers.values()))
        
        self.conversation_context = {
            "vehicle_info": vehicle_analysis["description"],
            "vehicle_value_category": vehicle_analysis["vehicle_category"],
            "premium_offered": None,
            "outcome": outcome,
            "negotiation_stage": 0,
            "turn_count": 0,
            "recommended_provider": provider
        }
        
        # 3. User Opening
        openings = [
            "Hi, I just bought this car. Need insurance.",
            "Can you help me insure my new vehicle?",
            "Looking for a quote for my new car."
        ]
        user_msg = f"{random.choice(openings)} [Vehicle Description: {vehicle_analysis['description']}]"
        turn_no = 1
        
        conversation_export.append((conversation_id, turn_no, "User", user_msg))
        self.conversation_history.append((conversation_id, turn_no, "User", user_msg))
        
        # LOOP
        max_turns = 28
        turn_no += 1
        
        while turn_no <= max_turns:
            self.conversation_context['turn_count'] = turn_no
            
            # -- AGENT TURN --
            if turn_no == 2:
                # Removed image_url arg
                strategy, agent_resp = self.generate_agent_response(user_msg, self.conversation_context, self.conversation_history, 
                                                                    conversation_id, turn_no, recommended_provider=provider)
            else:
                strategy, agent_resp = self.generate_agent_response(user_msg, self.conversation_context, self.conversation_history, 
                                                                    conversation_id, turn_no, recommended_provider=provider)
            
            conversation_export.append((conversation_id, turn_no, f"Agent ({strategy})", agent_resp))
            self.conversation_history.append((conversation_id, turn_no, "Agent", agent_resp))
            turn_no += 1
            
            if self.conversation_context.get('finalized'): break
            if turn_no > max_turns: break
            
            # -- USER TURN --
            intent, user_resp = self.generate_user_utterance(agent_resp, self.conversation_context, self.conversation_history, persona)
            
            conversation_export.append((conversation_id, turn_no, f"User ({intent})", user_resp))
            self.conversation_history.append((conversation_id, turn_no, "User", user_resp))
            user_msg = user_resp # Update for next loop
            turn_no += 1
            
            if intent in ["Confirm Plan", "Reject Offer"]:
                self.conversation_context['finalized'] = True
                # One final agent closing
                closing = "Great, I'll process that right away." if intent == "Confirm Plan" else "Understood, have a good day."
                conversation_export.append((conversation_id, turn_no, "Agent (Default)", closing))
                break
                
        return conversation_export, self.thinking_log

# --- 5. Main Execution Loop ---

def main():
    # --- CONFIGURATION ---
    TOTAL_CONVERSATIONS = 5  # <-- SET THIS TO DESIRED NUMBER (2, 5, 10, etc.)
    # ---------------------

    print("--- Memory Agent Conversation Generator ---")

    # 1. Load Data
    try:
        with open("motor-insurance-updated.json", "r", encoding="utf-8") as f:
            insurance_data = json.load(f)
    except Exception as e:
        print(f"Error loading motor-insurance-updated.json: {e}")
        return

    # 2. Generate or Load Personas (Caching implemented)
    print(f"Checking for existing personas...")
    personas_data = []
    
    if os.path.exists("personas.json"):
        print("Found personas.json, loading...")
        try:
            with open("personas.json", "r", encoding="utf-8") as f:
                personas_data = json.load(f) # Load as list of dicts
            print(f"Loaded {len(personas_data)} personas from cache.")
        except Exception as e:
            print(f"Error loading personas.json: {e}. Will regenerate.")
            personas_data = []
    
    # If we need more personas than we have, generate them
    if len(personas_data) < TOTAL_CONVERSATIONS:
        needed = TOTAL_CONVERSATIONS - len(personas_data)
        print(f"Need {needed} more personas. Generating...")
        
        # Logic to generate efficiently
        target_archetypes = ARCHETYPES
        n_per = 1
        
        if needed <= 3:
            target_archetypes = ARCHETYPES[:needed]
            n_per = 1
        else:
            n_per = math.ceil(needed / 10) # 10 archetypes
            
        new_personas = generate_personas(n_per_archetype=n_per, archetypes_list=target_archetypes)
        personas_data.extend(new_personas)
        
        # Save updated list
        try:
            with open("personas.json", "w", encoding="utf-8") as f:
                json.dump(personas_data, f, indent=4)
            print("Saved updated personas to personas.json")
        except Exception as e:
            print(f"Error saving personas.json: {e}")

    # Trim to exact count
    personas_data = personas_data[:TOTAL_CONVERSATIONS]
    
    # 3. Initialize Generator
    generator = DialogueGenerator(insurance_data)
    
    # 4. Generate Conversations
    print(f"Starting Generation of {len(personas_data)} Conversations...")
    
    # Prepare CSV headers
    with open("conversations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_id", "turn_no", "speaker", "utterance"])
        
    with open("thinking.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_id", "turn_no", "agent_name", "thinking"])
    
    # Loop
    for i in tqdm(range(len(personas_data))):
        persona = personas_data[i]
        conversation_id = i + 1
        
        try:
            conv, think = generator.generate_conversation(conversation_id, persona)
            
            # Save
            with open("conversations.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(conv)
                
            with open("thinking.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for t in think:
                    writer.writerow([t['conversation_id'], t['turn_no'], t['agent_name'], t['thinking']])
            
        except Exception as e:
            print(f"Error in ID {conversation_id}: {e}")
            import traceback
            traceback.print_exc()
            
    print("Done! Saved to conversations.csv and thinking.csv")

if __name__ == "__main__":
    main()
