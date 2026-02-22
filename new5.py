import json
import random
import csv
import re
import requests
from PIL import Image
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import pandas as pd

# Load the VLM model and processor
print("Loading VLM model and processor...")
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-3-4b-it",
    device_map="auto"
)
print("+-+-+-+-+-+-+-+-+")
print(f"Using device: {next(model.parameters()).device}")
print("+-+-+-+-+-+-+-+-+")




def gen_utterance(prompt_, temperature=0.7, max_len=800, image_url=None):
    """
    Generates utterances using Gemma model.
    If image_url is provided, sends multimodal prompt.
    Uses processor for tokenization for both multimodal and text-only inputs.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"} if image_url else None,
                {"type": "text", "text": prompt_}
            ]
        }
    ]
    messages[0]["content"] = [item for item in messages[0]["content"] if item is not None]

    raw_image = None
    if image_url:
        
        # Define a browser-like User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # --- END OF FIX ---

        try:
            # Add the 'headers=headers' argument to your request
            response = requests.get(image_url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            raw_image = Image.open(response.raw).convert("RGB")
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Warning: Could not process image URL {image_url}. Error: {e}. Proceeding with text only.")
            # Log inaccessible URL
            with open("inaccessible_urls.txt", "a", encoding="utf-8") as f:
                f.write(f"{image_url} - Error: {str(e)}\n")
            messages[0]["content"] = [item for item in messages[0]["content"] if item['type'] != 'image']
            image_url = None

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text_prompt,
        images=raw_image,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_len, temperature=temperature, do_sample=True)
    response_start_index = inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][response_start_index:], skip_special_tokens=True).strip()

    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0].strip()
    if "User:" in response:
        response = response.split("User:")[-1].strip()

    return response











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
        """Engagement Expert: Analyzes rapport-building opportunities"""
        prompt = f"""You are an Engagement Expert. Analyze briefly in ONE short sentence.

User: "{user_message}"
Turn: {len(history)}

Provide ONE brief sentence about engagement strategy (tone, rapport, timing)."""

        return gen_utterance(prompt, temperature=0.6, max_len=60)
    
    def _keyterm_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Keyterm Expert: Identifies key features to highlight"""
        prompt = f"""You are a Keyterm Expert. Identify key points briefly.

User: "{user_message}"
Vehicle: {context.get('vehicle_value_category', 'unknown')}

ONE brief sentence identifying key features to emphasize."""

        return gen_utterance(prompt, temperature=0.6, max_len=60)
    
    def _intent_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Intent Expert: Identifies user intent"""
        prompt = f"""You are an Intent Expert. Classify intent briefly.

User: "{user_message}"

Classify as: Request quote, Ask coverage, Express concern, Request info, Negotiate price, or Confirm/Reject.
ONE brief sentence only."""

        return gen_utterance(prompt, temperature=0.5, max_len=50)
    
    def _sentiment_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Sentiment Expert: Analyzes emotional tone"""
        prompt = f"""You are a Sentiment Expert. Analyze tone briefly.

User: "{user_message}"

ONE brief sentence about emotional state and response approach."""

        return gen_utterance(prompt, temperature=0.6, max_len=50)
    
    def _knowledge_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Knowledge Expert: Determines what information to retrieve"""
        prompt = f"""You are a Knowledge Expert. Identify needed information briefly.

User: "{user_message}"
Provider: {context.get('recommended_provider', 'unknown')}

ONE brief sentence about what knowledge to use."""

        return gen_utterance(prompt, temperature=0.6, max_len=50)
    
    def orchestrate(self, context: Dict, user_message: str, history: List, 
                   active_experts: List[str] = None) -> Dict[str, str]:
        """
        Orchestrate multiple experts to produce reasoning
        Returns dict of expert_name -> thinking
        """
        if active_experts is None:
            # Select relevant experts based on conversation stage
            active_experts = self._select_experts(context, user_message, len(history))
        
        thinking = {}
        for expert_name in active_experts:
            if expert_name in self.experts:
                try:
                    reasoning = self.experts[expert_name](context, user_message, history)
                    thinking[expert_name] = reasoning
                except Exception as e:
                    print(f"Error in {expert_name}: {e}")
                    thinking[expert_name] = f"Unable to generate reasoning: {str(e)}"
        
        return thinking
    
    def _select_experts(self, context: Dict, user_message: str, turn_count: int) -> List[str]:
        """Dynamically select which experts to invoke based on context"""
        experts = ["Engagement Expert"]  # Always active
        
        msg_lower = user_message.lower()
        
        # First turn - need everything
        if turn_count <= 2:
            return ["Engagement Expert", "Keyterm Expert", "Knowledge Expert"]
        
        # Intent detection always useful
        experts.append("Intent Expert")
        
        # Sentiment for concerns or negotiations
        if any(word in msg_lower for word in ["concern", "worried", "not sure", "expensive", "discount"]):
            experts.append("Sentiment Expert")
        
        # Keyterm when discussing features
        if any(word in msg_lower for word in ["feature", "cover", "include", "benefit", "what", "how"]):
            experts.append("Keyterm Expert")
        
        # Knowledge when asking specifics
        if any(word in msg_lower for word in ["price", "cost", "detail", "information", "tell me", "explain"]):
            experts.append("Knowledge Expert")
        
        # Limit to 4 experts per turn for efficiency
        return experts[:4]
    
    def synthesize(self, thinking: Dict[str, str], context: Dict) -> str:
        """Synthesize expert thinking into orchestrator summary"""
        experts_summary = "\n".join([f"{name}: {thought}" for name, thought in thinking.items()])
        
        prompt = f"""Synthesize expert insights into ONE brief sentence.

Expert Analysis:
{experts_summary}

ONE sentence overall strategy."""

        return gen_utterance(prompt, temperature=0.6, max_len=50)











class DialogueGenerator:
    def __init__(self, insurance_data):
        self.insurance_data = insurance_data
        self.conversation_context = {}
        self.conversation_history = []
        self.intent_history = []
        self.strategy_history = []
        self.thinking_log = []  # New: store all thinking records
        
        # Initialize expert system
        self.experts = ExpertAgents()
        
        # Pre-process insurer names
        self.insurers = {name.lower(): name for name in self.insurance_data["Motor Insurance"]["description"].keys()}
        
        # Define agent strategies
        self.agent_strategies = {
            "Default": "Provide a standard, professional, and helpful response.",
            "Credibility": "Build trust by highlighting company reputation, stats, and awards.",
            "Emotional": "Appeal to the user's sense of security, safety, and peace of mind.",
            "Logical": "Use facts, numbers, and rational arguments to explain benefits.",
            "Personal": "Tailor the response to the user's specific situation and expressed needs.",
            "Persona": "Adopt an expert persona, using personal experience to guide the user."
        }
        
        # Expanded and varied user openings
        self.user_openings = [
            "Hi, I just bought a new car. Can you help me find a good insurance policy?",
            "Hello, I need to get insurance for my new vehicle. What do you recommend?",
            "I'm looking for an insurance plan for my brand new car.",
            "Hey, I need to sort out insurance for my new car. Can you help?",
            "Just got my first car! Need help with insurance.",
            "I bought a car yesterday and need to insure it quickly.",
            "Looking for comprehensive coverage for my new vehicle.",
            "Can you suggest insurance options for my newly purchased car?",
            "Need motor insurance urgently. Just bought this beauty!",
            "Hi! New car owner here. What insurance should I get?"
        ]

        # Intent transitions - what intents naturally follow others
        self.intent_transitions = {
            "Initial": ["Ask Coverage Details", "Request quote", "Request additional info"],
            "Request quote": ["Negotiate price", "Ask Coverage Details", "Express Concern"],
            "Ask Coverage Details": ["Express Concern", "Request quote", "Request additional info"],
            "Express Concern": ["Request additional info", "Negotiate price", "Ask Coverage Details"],
            "Request additional info": ["Request quote", "Express Concern", "Negotiate price"],
            "Ask price or premium": ["Negotiate price", "Express Concern", "Request additional info"],
            "Negotiate price": ["Confirm Plan", "Express Concern", "Request additional info", "Reject Offer"]
        }

        # Strategy transitions - what strategies work well after others
        self.strategy_transitions = {
            "Default": ["Credibility", "Logical", "Personal"],
            "Credibility": ["Logical", "Emotional", "Personal"],
            "Emotional": ["Personal", "Logical", "Credibility"],
            "Logical": ["Personal", "Emotional", "Credibility"],
            "Personal": ["Emotional", "Logical", "Default"],
            "Persona": ["Credibility", "Personal", "Emotional"]
        }

        # Varied user personalities
        self.user_personalities = [
            "budget-conscious",
            "detail-oriented", 
            "skeptical",
            "enthusiastic",
            "cautious",
            "tech-savvy",
            "first-time buyer",
            "experienced buyer"
        ]

        # Context-specific responses for more natural flow
        self.contextual_templates = {
            "budget-conscious": {
                "Negotiate price": [
                    "That seems quite expensive for my budget. Can we work something out?",
                    "I was hoping for something more affordable. Any discounts available?",
                    "Is there a way to reduce this premium? It's a bit steep for me."
                ]
            },
            "detail-oriented": {
                "Ask Coverage Details": [
                    "Could you elaborate on what exactly is covered under collision damage?",
                    "I'd like to understand the claim process in detail.",
                    "What are the specific exclusions I should be aware of?"
                ]
            },
            "skeptical": {
                "Express Concern": [
                    "I've heard mixed reviews about insurance claims. How reliable is your process?",
                    "What guarantees do I have that claims will be processed quickly?",
                    "How do I know I won't face issues during claim settlement?"
                ]
            }
        }
        
        # Reasonable deductible ranges in INR based on vehicle category
        self.deductible_ranges = {
            "Economy": {"min": 1000, "standard": 2500, "max": 5000},
            "Mid-range": {"min": 2500, "standard": 5000, "max": 10000},
            "Premium": {"min": 5000, "standard": 10000, "max": 25000}
        }

    def _extract_price(self, text: str) -> Optional[int]:
        """Extracts the first numerical price from a string."""
        match = re.search(r'₹\s*([\d,]+)|([\d,]+)\s*rs', text, re.IGNORECASE)
        if match:
            price_str = (match.group(1) or match.group(2)).replace(',', '')
            return int(price_str)
        return None

    def _retrieve_relevant_info(self, query: str, recommended_provider: str = None) -> Dict:
        """Retrieves specific, relevant information from the knowledge base."""
        query_lower = query.lower()
        data = self.insurance_data["Motor Insurance"]
        
        # Prioritize recommended provider if specified 
        if recommended_provider:
            for name_lower, original_name in self.insurers.items():
                if recommended_provider.lower() in name_lower or name_lower in recommended_provider.lower():
                    return {
                        f"Info for {original_name}": {
                            "description": data["description"].get(original_name),
                            "pricing": data["pricing_information"]["comprehensive_premium_ranges"].get(original_name),
                            "unique_features": data["insurer_specific_features"].get(original_name)
                        }
                    }
        
        # Check for specific insurers
        for name_lower, original_name in self.insurers.items():
            if name_lower in query_lower or original_name.split(' ')[0].lower() in query_lower.split(' '):
                return {
                    f"Info for {original_name}": {
                        "description": data["description"].get(original_name),
                        "pricing": data["pricing_information"]["comprehensive_premium_ranges"].get(original_name),
                        "unique_features": data["insurer_specific_features"].get(original_name)
                    }
                }

        # Check for add-ons
        addon_keywords = ['add-on', 'addon', 'rider', 'zero depreciation', 'engine protection', 
                         'return to invoice', 'consumables', 'roadside assistance', 'tyre protection']
        if any(keyword in query_lower for keyword in addon_keywords):
            return {"Relevant Add-Ons": data["add_ons_detailed"]}

        # Check for pricing
        price_keywords = ['price', 'premium', 'cost', 'quote', 'rate', 'how much', 'expensive', 'cheap', 'affordable']
        if any(keyword in query_lower for keyword in price_keywords):
            return {"Pricing Information": data["pricing_information"]["comprehensive_premium_ranges"]}

        # Check for features
        feature_keywords = ['feature', 'coverage', 'cover', 'benefit', 'protect', 'what is covered', 'include']
        if any(keyword in query_lower for keyword in feature_keywords):
            return {"Coverage Features": data["features_classification"]}

        # Fallback
        return {
            "Core Coverage": data["features_classification"]["core_coverage_features"],
            "Service Features": data["features_classification"]["service_features"][:3]
        }

    def _get_next_strategy(self, last_strategy: str = None) -> str:
        """Select next strategy based on conversation flow."""
        if not last_strategy or last_strategy not in self.strategy_transitions:
            return random.choice(list(self.agent_strategies.keys()))
        
        # Avoid repeating same strategy
        possible_strategies = [s for s in self.strategy_transitions[last_strategy] if s not in self.strategy_history[-2:]]
        if not possible_strategies:
            possible_strategies = self.strategy_transitions[last_strategy]
        
        return random.choice(possible_strategies)


    def _get_next_intent(self, last_intent: str, personality: str, context: Dict) -> str:
        """Select next user intent based on conversation flow and personality."""
        
        # Get current turn count
        turn_count = context.get('turn_count', 0)
        
        # CRITICAL: Check if already finalized - prevent duplicate finalization
        if "Confirm Plan" in self.intent_history or "Reject Offer" in self.intent_history:
            # Already finalized - should not generate more intents
            return "Request additional info"  # Fallback (shouldn't be used)
        
        # CRITICAL: Block finalization BEFORE turn 34
        if turn_count < 34:
            # Filter out ALL finalization intents before turn 34
            possible_intents = []
            
            # Check if we should move to negotiation phase (but not finalization)
            if context.get('premium_offered') and not context.get('negotiation_started'):
                if personality in ["budget-conscious", "skeptical"]:
                    return "Negotiate price"
            
            # Use natural transitions (excluding finalization)
            if last_intent in self.intent_transitions:
                possible_intents = [i for i in self.intent_transitions[last_intent] 
                                  if i not in ["Confirm Plan", "Reject Offer"]]
                
                # Avoid repeating recent intents
                filtered = [i for i in possible_intents if i not in self.intent_history[-2:]]
                if filtered:
                    return random.choice(filtered)
            
            # Fallback (no finalization)
            return random.choice(["Ask Coverage Details", "Express Concern", "Request additional info"])
        
        # AFTER turn 34: FORCE finalization if not done yet
        if turn_count >= 34:
            # Check if premium was offered
            premium_offered = context.get('premium_offered')
            
            if not premium_offered:
                # No premium offered yet - ask for it one more time
                return "Request quote"
            
            # Premium exists - make final decision based on outcome
            if context['outcome'] == 'Accept':
                return "Confirm Plan"
            else:
                return "Reject Offer"
        
        # This shouldn't be reached, but safety fallback
        return random.choice(["Ask Coverage Details", "Express Concern", "Request additional info"])

    
    def generate_agent_response(self, user_message: str, context: Dict, conversation_history: List, 
                               conversation_id: int, turn_no: int,
                               image_url: str = None, recommended_provider: str = None) -> Tuple[str, str]:
        
        # MULTI-AGENT REASONING - Invoke expert agents
        expert_thinking = self.experts.orchestrate(
            context=context,
            user_message=user_message,
            history=conversation_history
        )
        
        # Log thinking for each expert
        for expert_name, thinking in expert_thinking.items():
            self.thinking_log.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "agent_name": expert_name,
                "thinking": thinking
            })
        
        # Synthesize expert insights
        orchestrator_summary = self.experts.synthesize(expert_thinking, context)
        self.thinking_log.append({
            "conversation_id": conversation_id,
            "turn_no": turn_no,
            "agent_name": "Orchestrator",
            "thinking": orchestrator_summary
        })
        
        # Build conversation history
        history_context = "\n".join([f"{speaker}: {msg}" for _, _, speaker, msg in conversation_history[-6:]])
        
        # Get strategy
        last_strategy = self.strategy_history[-1] if self.strategy_history else None
        chosen_strategy = self._get_next_strategy(last_strategy)
        self.strategy_history.append(chosen_strategy)
        
        # Strategy descriptions
        strategy_descriptions = {
            "Default": "Be professional and informative.",
            "Credibility": "Emphasize company reputation, awards, claim settlement ratio.",
            "Emotional": "Appeal to security, family safety, peace of mind.",
            "Logical": "Use statistics, comparisons, cost-benefit analysis.",
            "Personal": "Relate to their specific situation and vehicle.",
            "Persona": "Share a relevant anecdote or personal experience."
        }
        
        is_first_agent_turn = len(conversation_history) == 1
        
        # Get deductible info for current vehicle category
        vehicle_category = context.get('vehicle_value_category', 'Mid-range')
        deductibles = self.deductible_ranges[vehicle_category]
        
        if is_first_agent_turn:
            provider_info = self._retrieve_relevant_info("", recommended_provider)
            insurance_context_str = json.dumps(provider_info, indent=2)
            
            action_instruction = f"""
            This is your first response. You must:
            1. Acknowledge their new car purchase enthusiastically
            2. Recommend {recommended_provider} specifically
            3. Mention 2-3 key benefits relevant to their vehicle type
            4. Keep it conversational and under 4 sentences
            """
        else:
            relevant_kb = self._retrieve_relevant_info(user_message, recommended_provider)
            insurance_context_str = json.dumps(relevant_kb, indent=2)
            
            # Check for deductible queries
            if "deductible" in user_message.lower():
                action_instruction = f"""Explain deductible options for {vehicle_category} vehicles:
                - Standard deductible: ₹{deductibles['standard']:,}
                - Lower option available: ₹{deductibles['min']:,}
                - Higher option: ₹{deductibles['max']:,}
                Use RUPEES (₹), not dollars. Explain how deductible affects premium."""
                
            # Dynamic action instructions based on context
            elif "price" in user_message.lower() and not context.get('premium_offered'):
                base_premium = {'Economy': 15000, 'Mid-range': 25000, 'Premium': 40000}[context['vehicle_value_category']]
                offer_price = int(base_premium * random.uniform(1.05, 1.25))
                self.conversation_context['premium_offered'] = offer_price
                action_instruction = f"Quote an annual premium of ₹{offer_price:,}. Briefly mention what's included. Use RUPEES (₹), not dollars."
            
            # IMPORTANT: Handle negotiation differently
            elif "negotiate" in user_message.lower() or "discount" in user_message.lower() or "expensive" in user_message.lower() or "reduce" in user_message.lower():
                if context.get('negotiation_stage', 0) == 0:
                    # FIRST negotiation attempt - justify price with features
                    self.conversation_context['negotiation_stage'] = 1
                    self.conversation_context['negotiation_started'] = True
                    action_instruction = f"""User wants a discount. You must FIRST:
                    1. Justify the current premium of ₹{context.get('premium_offered', 25000):,}
                    2. List specific features they get: zero depreciation, roadside assistance, cashless network
                    3. Mention claim settlement ratio or other credibility factors
                    4. DO NOT offer discount yet - explain value first
                    Use RUPEES (₹), not dollars."""
                else:
                    # Offer discount
                    current = context['premium_offered']
                    new_price = int(current * 0.92)
                    self.conversation_context['premium_offered'] = new_price
                    self.conversation_context['negotiation_stage'] += 1
                    action_instruction = f"""User wants discount. Now offer:
                    1. Acknowledge their concern
                    2. Offer discounted price of ₹{new_price:,} (8% off)
                    3. Mention this is best offer you can provide
                    Use RUPEES (₹), not dollars."""
            else:
                action_instruction = f"Respond using {chosen_strategy} strategy. Address their specific question/concern. Always use RUPEES (₹) for any monetary values, never dollars."

        # Include expert insights in prompt
        expert_summary = "\n".join([f"{name}: {thought[:60]}..." for name, thought in expert_thinking.items()])

        prompt = f"""
You are an experienced insurance agent having a natural conversation. Be concise (3-4 sentences max), conversational, and human-like.

CRITICAL: Always use Indian Rupees (₹) for ALL monetary values. NEVER use dollars ($) or any other currency.

Expert Analysis:
{expert_summary}

Orchestrator: {orchestrator_summary[:80]}

Conversation History:
{history_context}

Current Message: "{user_message}"

{action_instruction}

Strategy: {strategy_descriptions[chosen_strategy]}

Relevant Knowledge:
{insurance_context_str}

Vehicle Category: {vehicle_category}

Respond naturally without mentioning your strategy or experts. Vary your language and avoid repetitive phrases.
Remember: ALWAYS use ₹ for prices, NEVER use $.
"""

        try:
            response = gen_utterance(
                prompt_=prompt,
                temperature=0.8,
                max_len=350,
                image_url=image_url if is_first_agent_turn else None
            )
            
            # Log final synthesizer thinking
            self.thinking_log.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "agent_name": "Synthesizer",
                "thinking": f"Generated response using {chosen_strategy} based on expert insights."
            })
            
            # Extract price if mentioned
            price_in_response = self._extract_price(response)
            if price_in_response:
                self.conversation_context['premium_offered'] = price_in_response
            
            return chosen_strategy, response
            
        except Exception as e:
            print(f"Error generating agent response: {e}")
            return "Default", "Let me help you find the right coverage for your new car. What's most important to you?"

    def generate_user_utterance(self, agent_message: str, context: Dict, conversation_history: List, 
                               personality: str) -> Tuple[str, str]:
        
        # Build conversation history
        history_context = "\n".join([f"{speaker}: {msg}" for _, _, speaker, msg in conversation_history[-5:]])
        
        # Get intent
        last_intent = self.intent_history[-1] if self.intent_history else "Initial"
        chosen_intent = self._get_next_intent(last_intent, personality, context)
        self.intent_history.append(chosen_intent)
        
        # Personality-specific instructions
        personality_traits = {
            "budget-conscious": "You're very price-sensitive and looking for value.",
            "detail-oriented": "You want to understand everything thoroughly.",
            "skeptical": "You're cautious and need convincing.",
            "enthusiastic": "You're excited about your new car and open to suggestions.",
            "cautious": "You're careful and want to avoid any risks.",
            "tech-savvy": "You value digital features and convenience.",
            "first-time buyer": "You're new to this and have questions.",
            "experienced buyer": "You know what you want and can compare options."
        }
        
        # Check if we should finalize
        is_final_turn = context.get('negotiation_stage', 0) >= 2 and context.get('premium_offered')
        
        if is_final_turn:
            if context['outcome'] == 'Accept':
                instruction = f"Accept the offer of ₹{context['premium_offered']:,}. Ask about next steps."
                chosen_intent = "Confirm Plan"
            else:
                instruction = f"Politely decline. The ₹{context['premium_offered']:,} is over your ₹{context['user_budget']:,} budget."
                chosen_intent = "Reject Offer"
        else:
            # Use contextual templates if available
            if personality in self.contextual_templates and chosen_intent in self.contextual_templates[personality]:
                template_hint = random.choice(self.contextual_templates[personality][chosen_intent])
                instruction = f"Respond with intent '{chosen_intent}'. Consider: {template_hint}"
            else:
                instruction = f"Respond with intent '{chosen_intent}'. Be {personality}."

        prompt = f"""
You are a {personality} car owner in a conversation with an insurance agent.

CRITICAL: Always use Indian Rupees (₹) for ALL monetary values when discussing prices.

Recent Conversation:
{history_context}

Personality: {personality_traits.get(personality, "Be yourself")}
Task: {instruction}

Respond naturally in 1-2 sentences. Don't repeat phrases from the conversation. Be conversational and authentic.
Use ₹ for any price mentions, not $.
"""

        try:
            response = gen_utterance(
                prompt_=prompt,
                temperature=0.9,
                max_len=150
            )
            
            return chosen_intent, response
            
        except Exception as e:
            print(f"Error generating user response: {e}")
            fallback_responses = {
                "Request quote": "What's the price for that?",
                "Ask Coverage Details": "What does this cover exactly?",
                "Express Concern": "I'm not sure about this.",
                "Negotiate price": "Can you do better on the price?"
            }
            return chosen_intent, fallback_responses.get(chosen_intent, "Tell me more.")

    def analyze_vehicle_with_vlm(self, image_url: str) -> Dict:
        """Analyze vehicle using VLM."""
        try:
            prompt = """Look at this vehicle and determine:
1. Category: Is it an Economy (basic/budget), Mid-range (standard family car), or Premium (luxury/sports) vehicle?
2. Type: What kind of vehicle is it? (SUV, Sedan, Hatchback, etc.)
3. Notable features visible

Be specific and concise."""

            print("Analyzing vehicle...")
            analysis = gen_utterance(
                prompt_=prompt,
                temperature=0.3,
                max_len=200,
                image_url=image_url
            )
            
            # Parse category
            analysis_lower = analysis.lower()
            if "premium" in analysis_lower or "luxury" in analysis_lower or "sports" in analysis_lower:
                category = "Premium"
            elif "economy" in analysis_lower or "budget" in analysis_lower or "basic" in analysis_lower:
                category = "Economy"
            else:
                category = "Mid-range"
            
            return {
                "vehicle_category": category,
                "description": analysis
            }
            
        except Exception as e:
            print(f"VLM analysis error: {e}")
            return {
                "vehicle_category": random.choice(['Economy', 'Mid-range', 'Premium']),
                "description": "Standard vehicle requiring comprehensive coverage."
            }

    def save_conversation(self, conversation: List[Tuple], append_mode: bool = False):
        """Save conversation to CSV file"""
        output_file = "conversations.csv"
        mode = 'a' if append_mode else 'w'
        
        with open(output_file, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not append_mode:
                writer.writerow(["conversation_id", "turn_no", "speaker", "utterance"])
            for row in conversation:
                writer.writerow(row)
    
    def save_thinking(self, thinking_records: List[Dict], append_mode: bool = False):
        """Save thinking logs to CSV file"""
        thinking_file = "thinking.csv"
        mode = 'a' if append_mode else 'w'
        
        with open(thinking_file, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not append_mode:
                writer.writerow(["conversation_id", "turn_no", "agent_name", "thinking"])
            for record in thinking_records:
                writer.writerow([
                    record["conversation_id"],
                    record["turn_no"],
                    record["agent_name"],
                    record["thinking"]
                ])

    def generate_conversation(self, image_url: str, conversation_id: int) -> Tuple[List[Tuple], List[Dict]]:
        """Generate a complete conversation. Returns (conversation, thinking_log)"""
        conversation = []
        self.conversation_history = []
        self.intent_history = []
        self.strategy_history = []
        conversation_thinking = []
        
        # Analyze vehicle
        vehicle_analysis = self.analyze_vehicle_with_vlm(image_url)
        
        # Select personality for this conversation
        personality = random.choice(self.user_personalities)
        
        # Select random insurer to recommend
        all_providers = list(self.insurance_data["Motor Insurance"]["description"].keys())
        recommended_provider = random.choice(all_providers)
        
        # Initialize context
        base_premiums = {'Economy': 15000, 'Mid-range': 25000, 'Premium': 40000}
        base_premium = base_premiums[vehicle_analysis['vehicle_category']]
        
        # Determine outcome - 75% accept, 25% reject
        if random.random() < 0.25:
            outcome = 'Reject'
            user_budget = int(base_premium * random.uniform(0.75, 0.90))
        else:
            outcome = 'Accept'
            user_budget = int(base_premium * random.uniform(0.95, 1.20))
        
        self.conversation_context = {
            "vehicle_info": vehicle_analysis["description"],
            "vehicle_value_category": vehicle_analysis["vehicle_category"],
            "premium_offered": None,
            "negotiation_stage": 0,
            "negotiation_started": False,
            "outcome": outcome,
            "user_budget": user_budget,
            "personality": personality,
            "recommended_provider": recommended_provider,
            "turn_count": 0,
            "recent_intents": [],
            "finalized": False
        }
        
        print(f"\n=== Conversation {conversation_id} ===")
        print(f"Personality: {personality} | Outcome: {outcome} | Provider: {recommended_provider}")
        print(f"Vehicle: {vehicle_analysis['vehicle_category']} | Budget: ₹{user_budget:,}")
        
        # Opening
        turn_no = 1
        opening = random.choice(self.user_openings)
        
        # Vary how the image is referenced
        image_refs = [
            f"Here's a picture of my new car: {image_url}",
            f"Check out my new ride: {image_url}",
            f"This is what I just bought: {image_url}",
            f"Take a look: {image_url}",
            f"[Shares image: {image_url}]"
        ]
        
        user_msg = f"{opening} {random.choice(image_refs)}"
        conversation.append((conversation_id, turn_no, "User", user_msg))
        self.conversation_history.append((conversation_id, turn_no, "User", user_msg))
        turn_no += 1
        
        # Generate conversation - MAX 40 TURNS (to allow finalization after turn 34)
        max_turns = 40
        
        while turn_no <= max_turns:
            self.conversation_context['turn_count'] = turn_no
            
            # Check if already finalized BEFORE generating new responses
            if self.conversation_context.get('finalized'):
                print("Conversation already finalized. Ending.")
                break
            
            # Clear thinking log before generating new responses
            self.thinking_log = []
            
            # Agent turn
            if turn_no == 2:
                # First agent response with image
                strategy, response = self.generate_agent_response(
                    self.conversation_history[-1][3],
                    self.conversation_context,
                    self.conversation_history,
                    conversation_id,
                    turn_no,
                    image_url=image_url,
                    recommended_provider=recommended_provider
                )
            else:
                strategy, response = self.generate_agent_response(
                    self.conversation_history[-1][3],
                    self.conversation_context,
                    self.conversation_history,
                    conversation_id,
                    turn_no,
                    recommended_provider=recommended_provider
                )
            
            # Add to conversation with strategy annotation
            conversation.append((conversation_id, turn_no, f"Agent ({strategy})", response))
            self.conversation_history.append((conversation_id, turn_no, "Agent", response))
            
            # Store thinking records for this conversation
            conversation_thinking.extend(self.thinking_log)
            
            print(f"Turn {turn_no}: Agent ({strategy})")
            turn_no += 1
            
            # Check if max turns reached
            if turn_no > max_turns:
                break
            
            # Check if already finalized AGAIN before user turn
            if self.conversation_context.get('finalized'):
                print("Conversation finalized. Ending.")
                break
            
            # User turn
            intent, response = self.generate_user_utterance(
                self.conversation_history[-1][3],
                self.conversation_context,
                self.conversation_history,
                personality
            )
            
            # Add to conversation with intent annotation
            conversation.append((conversation_id, turn_no, f"User ({intent})", response))
            self.conversation_history.append((conversation_id, turn_no, "User", response))
            self.conversation_context['recent_intents'] = self.intent_history[-3:]
            print(f"Turn {turn_no}: User ({intent})")
            turn_no += 1
            
            # Check for conversation end - mark as finalized
            if intent in ["Confirm Plan", "Reject Offer"]:
                self.conversation_context['finalized'] = True
                
                # Add final agent response
                if intent == "Confirm Plan":
                    final_response = random.choice([
                        "Excellent choice! I'll send you the policy documents right away. Welcome to our family!",
                        "Perfect! Let me process this for you. You'll receive everything via email shortly.",
                        "Great decision! Your car is now in safe hands. Documents coming your way!"
                    ])
                else:
                    final_response = random.choice([
                        "I understand. Feel free to reach out if you change your mind. Drive safely!",
                        "No problem at all. We're here whenever you're ready. Best of luck!",
                        "That's okay. Take your time to decide. Our offer stands when you need us."
                    ])
                
                conversation.append((conversation_id, turn_no, "Agent (Default)", final_response))
                
                # Log final thinking
                conversation_thinking.append({
                    "conversation_id": conversation_id,
                    "turn_no": turn_no,
                    "agent_name": "Synthesizer",
                    "thinking": f"Conversation concluded with {intent}. Providing closing statement."
                })
                
                print(f"Turn {turn_no}: Agent (Default) - Final closing")
                break  # Exit immediately after finalization
        
        # If max turns reached without proper closure, force closure
        if not self.conversation_context.get('finalized') and conversation:
            if outcome == 'Accept':
                final_response = "Excellent! I'll finalize your policy. You'll receive confirmation shortly. Thank you for choosing us!"
            else:
                final_response = "I appreciate your time. If you'd like to reconsider, feel free to contact us anytime. Drive safe!"
            
            conversation.append((conversation_id, turn_no, "Agent (Default)", final_response))
            conversation_thinking.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "agent_name": "Synthesizer",
                "thinking": "Max turns reached. Providing forced closure."
            })
            print(f"Turn {turn_no}: Agent (Default) - Forced closing")
        
        return conversation, conversation_thinking

def main():
    # Load data
    try:
        with open("motor-insurance-updated.json", "r", encoding="utf-8") as f:
            insurance_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'motor-insurance-updated.json' not found.")
        return
    
    try:
        with open("urls.json", "r", encoding="utf-8") as f:
            image_links = json.load(f)
    except FileNotFoundError:
        print("Error: 'urls.json' not found.")
        return
    
    # Initialize generator
    generator = DialogueGenerator(insurance_data)
    
    # Clear inaccessible URLs file
    with open("inaccessible_urls.txt", "w", encoding="utf-8") as f:
        f.write("Inaccessible URLs Log\n")
        f.write("=" * 50 + "\n\n")
    
    # Generate conversations
    num_conversations = len(image_links)  # Adjust as needed
    
    print(f"Generating {num_conversations} conversations...")

    start_index = 600
    for i in tqdm(range(num_conversations)):
        image_url = image_links[i % len(image_links)]['src']
        conversation_id = i + 1
        
        try:
            # Generate conversation
            conversation, thinking = generator.generate_conversation(image_url, conversation_id)
            
            # Save immediately after generation (append mode after first conversation)
            append_mode = (i > 0)
            generator.save_conversation(conversation, append_mode=append_mode)
            generator.save_thinking(thinking, append_mode=append_mode)
            
            print(f"✓ Conversation {conversation_id} saved ({len(conversation)} turns)\n")
            
        except Exception as e:
            print(f"\n✗ Error in conversation {conversation_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Log the error and URL
            with open("inaccessible_urls.txt", "a", encoding="utf-8") as f:
                f.write(f"Conversation {conversation_id} failed: {image_url}\n")
                f.write(f"Error: {str(e)}\n\n")
            
            continue
    
    print(f"\n All conversations saved to conversations.csv")
    print(f" All thinking logs saved to thinking.csv")
    print(f" Inaccessible URLs logged to inaccessible_urls.txt")


def mainCont():
    # Load data
    try:
        with open("motor-insurance-updated.json", "r", encoding="utf-8") as f:
            insurance_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'motor-insurance-updated.json' not found.")
        return
    
    try:
        with open("urls.json", "r", encoding="utf-8") as f:
            image_links = json.load(f)
    except FileNotFoundError:
        print("Error: 'urls.json' not found.")
        return
    
    # Initialize generator
    generator = DialogueGenerator(insurance_data)
    
    # Copy old CSVs into new ones
    try:
        print("Copying old CSV data to new files...")
        conv_old = pd.read_csv("conversations.csv")
        think_old = pd.read_csv("thinking.csv")
        
        conv_old.to_csv("conversations_new.csv", index=False)
        think_old.to_csv("thinking_new.csv", index=False)
        
        start_index = len(conv_old['conversation_id'].unique())  # Automatically detect last ID
        print(f" Found {start_index} existing conversations. Continuing from {start_index + 1}...")
    except FileNotFoundError:
        print(" Old CSVs not found! Starting fresh.")
        start_index = 0
        pd.DataFrame().to_csv("conversations_new.csv", index=False)
        pd.DataFrame().to_csv("thinking_new.csv", index=False)

    # Clear inaccessible URLs file
    with open("inaccessible_urls.txt", "w", encoding="utf-8") as f:
        f.write("Inaccessible URLs Log (Continued)\n")
        f.write("=" * 50 + "\n\n")
    
    # Generate conversations
    num_conversations = len(image_links)
    print(f"Continuing generation from {start_index + 1} to {num_conversations}...")

    for i in tqdm(range(start_index, num_conversations)):
        image_url = image_links[i % len(image_links)]['src']
        conversation_id = i + 1
        
        try:
            # Generate conversation
            conversation, thinking = generator.generate_conversation(image_url, conversation_id)
            
            # Save immediately after generation (append mode after first)
            generator.save_conversation(conversation, filename="conversations_new.csv", append_mode=True)
            generator.save_thinking(thinking, filename="thinking_new.csv", append_mode=True)
            
            print(f"✓ Conversation {conversation_id} saved ({len(conversation)} turns)\n")
        
        except Exception as e:
            print(f"\n✗ Error in conversation {conversation_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Log the error and URL
            with open("inaccessible_urls.txt", "a", encoding="utf-8") as f:
                f.write(f"Conversation {conversation_id} failed: {image_url}\n")
                f.write(f"Error: {str(e)}\n\n")
            
            continue
    
    print(f"\n All conversations saved to conversations_new.csv")
    print(f" All thinking logs saved to thinking_new.csv")
    print(f" Inaccessible URLs logged to inaccessible_urls.txt")


if __name__ == "__main__":
    # mainCont()
    main()