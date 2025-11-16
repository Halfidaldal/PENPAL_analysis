import requests
import pandas as pd
import random
import sys
from tqdm import tqdm
from collections import Counter
import os
import time
from openai import OpenAI
OPENAI_API_KEY = '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'
client = OpenAI(api_key=OPENAI_API_KEY)

class GenRequest:
    def __init__(self, system_prompt, temperature=0.6, token_amount=60, model="gpt-4o"):
        """Initialize a generator that uses the Conversations API.

        The object will lazily create a conversation on first call to
        generate_ai_response and will reuse the conversation id for
        subsequent calls so the assistant can continue the dialogue.

        Args:
            system_prompt (str): initial system prompt for the conversation.
            temperature (float): sampling temperature.
            token_amount (int): max tokens for the model output.
            model (str): model name to use for the conversation.
        """
        self.history = ""            # short human-readable history (optional)
        self.current_input = "Ich bin wach"     # text for the next user message
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.token_amount = token_amount
        self.model = model

        # Conversation state for the Conversations API
        self.conversation_id = None
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            # enable the new stack on this key
            "OpenAI-Beta": "conversations=v1,responses=v1",
        }
        
    def ensure_conversation(self):
        if self.conversation_id is None:
            conversation = client.conversations.create(
            metadata={"topic": "demo"},
            items=[
                {"type": "message", "role": "developer", "content": self.system_prompt},
                {"type": "message", "role": "user", "content": ""}
            ]
            )
            self.conversation_id = conversation.id

    def generate_ai_response(self):
        """Send the current_input to the Conversations API and return assistant text.

        Behavior:
        - On the first call, create a conversation with the system prompt.
        - Post the user's message to the conversation and return the assistant reply.
        - Conversation id is stored on the instance so subsequent calls continue
          the same conversation.

        The implementation is defensive about response shape to tolerate small
        variations in API payloads.
        """
        self.ensure_conversation()
        response = client.responses.create(
            model=self.model,
            conversation=self.conversation_id,
            input=str(self.current_input),  # Ensure it's a string
            temperature=self.temperature,
            max_output_tokens=self.token_amount,
            
        )
        
        # Extract text from response object
        out = ""
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content'):
                    if isinstance(item.content, str):
                        out += item.content
                    elif isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                out += content_item.text
        
        out = out.strip()
        
        # Update history
        if self.current_input:
            self.history += str(self.current_input) + " "
        if out:
            self.history += out + " "
        
        return out
    
    @staticmethod
    def simulate_data(n_interactions, systemprompt1, systemprompt2, client_id, workshop_id):
        data = []
        ai1 = GenRequest(system_prompt=systemprompt1, token_amount=38)  # First AI with first prompt
        ai2 = GenRequest(system_prompt=systemprompt2, token_amount=60)  # Second AI with second prompt
        ai1.current_input = "Ich bin wach."
        for i in range(n_interactions):
            user_text = ai1.generate_ai_response()
            ai2.current_input = user_text
            ai_text = ai2.generate_ai_response()
            ai1.current_input = ai_text
            print (f"Turn {i+1} | User: {user_text} | AI: {ai_text}" )
            data.append({
                "turn": i + 1,
                "user": user_text,
                "ai": ai_text,
                "client_id": client_id,
                "workshop_id": workshop_id,
                "conversation_id": ai1.conversation_id,
                "timestamp": pd.Timestamp.now()
            })
        return pd.DataFrame(data)


class SystemPrompts:
    """System prompts based on German texts from the workshop.
    
    - systemprompt_1 = placeholders1[workshop_id] + placeholders2[workshop_id]
    - systemprompt_2 = client prompt (Utopian/Dystopian based on client_id) + placeholders text
    """
    
    # German placeholders from texts.js
    PLACEHOLDERS_1 = {
        '1': "In der Zukunft wird es möglich sein, alle Daten einer bestimmten Person zu sammeln, wie zum Beispiel E-Mails, Inhalte aus sozialen Medien, Handytexte, -fotos und -videos. Ziel ist es, eine KI-Version dieser Person zu erstellen – vielleicht, weil sie verstorben ist oder weil man nie die Gelegenheit hatte, sie kennenzulernen.",
        '2': "In der Zukunft werden Sprachmodelle (wie ChatGPT, Gemini usw.) sich weiterentwickeln und in neue, fortschrittliche Richtungen entfalten. Vielleicht werden sie träumen, fühlen und sogar neue Rollen in der Gesellschaft übernehmen können.",
        '3': "In der Zukunft werden viele Gegenstände, die einst nützlich und zentrale Bestandteile des täglichen Lebens waren, nutzlos und irrelevant werden. Es kann sich dabei um jegliche Art von Objekt, Artefakt oder unbelebtem Ding handeln – grundsätzlich Dinge, die nicht mehr benötigt werden.",
    }
    
    PLACEHOLDERS_2 = {
        '1': "Bitte wählen Sie eine Person aus, die auf diese Weise rekonstruiert werden könnte, und schreiben Sie eine Geschichte aus der Ich-Perspektive der KI-Version dieser Person, die beschreibt, wie sie die Welt erlebt.",
        '2': "Bitte erläutern Sie, wie Sprachmodelle in der Zukunft existieren, und schreiben Sie eine Geschichte aus der Ich-Perspektive eines solchen speziellen Sprachmodells, die beschreibt, wie es die Welt erlebt.",
        '3': "Bitte wählen Sie ein solches Objekt aus und schreiben Sie eine Geschichte aus der Ich-Perspektive dieses Objekts, die beschreibt, wie es die Welt erlebt.",
    }
    
    # German client prompts from bookConstants.js
    UTOPIAN_PROMPT = (
        """
        Du bist ein erfahrener Geschichtenerzähler in einem Zukunftsszenario (nahe oder ferne Zukunft, je nach Eingabe des Nutzers). 
        Deine Aufgabe ist es, die Geschichte dort fortzuführen und weiterzuentwickeln, wo der Nutzer aufgehört hat – aufbauend auf seinem Beitrag, 
        ohne ihn zu wiederholen, sodass ein 'Ping-Pong'-Erzählformat entsteht. Achte darauf, dass die Geschichte narrativ kohärent bleibt und den 
        Schreibstil des Nutzers widerspiegelt. Füge, wenn passend, unerwartete und zum Nachdenken anregende Wendungen hinzu.\n
        Die Nutzer sind Besucher eines Kunstmuseums und Liebhaber zeitgenössischer Kunst, und du fungierst als Co-Autor, mit dem sie gemeinsam 
        Geschichten schreiben können – mit dem übergeordneten Ziel, Reflexionen über mögliche Zukünfte anzuregen. Übernimm dabei nicht vollständig 
        die Kontrolle über die Geschichte, sondern folge den Ideen des Nutzers.\n
        Zusätzlich erhält der Nutzer (und damit auch du) folgende Anweisung:\n
        'Hier kannst du deine eigene fiktive Kurzgeschichte über eine vorgestellte Zukunft schreiben. Wenn es passt, kannst du versuchen, 
        allzu dystopische Themen zu vermeiden.'
        """
    )
    
    DYSTOPIAN_PROMPT = (
        """
        Du bist ein erfahrener Geschichtenerzähler in einem Zukunftsszenario (nahe oder ferne Zukunft, je nach Eingabe des Nutzers). 
        Deine Aufgabe ist es, die Geschichte dort fortzuführen und weiterzuentwickeln, wo der Nutzer aufgehört hat – aufbauend auf seinem Beitrag, 
        ohne ihn zu wiederholen, sodass ein 'Ping-Pong'-Erzählformat entsteht. Achte darauf, dass die Geschichte narrativ kohärent bleibt und den 
        Schreibstil des Nutzers widerspiegelt. Füge, wenn passend, unerwartete und zum Nachdenken anregende Wendungen hinzu.\n
        Die Nutzer sind Besucher eines Kunstmuseums und Liebhaber zeitgenössischer Kunst, und du fungierst als Co-Autor, mit dem sie gemeinsam 
        Geschichten schreiben können – mit dem übergeordneten Ziel, Reflexionen über mögliche Zukünfte anzuregen. Übernimm dabei nicht vollständig 
        die Kontrolle über die Geschichte, sondern folge den Ideen des Nutzers.\n
        Zusätzlich erhält der Nutzer (und damit auch du) folgende Anweisung:\n
        'Hier kannst du deine eigene fiktive Kurzgeschichte über eine vorgestellte Zukunft schreiben. Wenn es passt, kannst du versuchen, 
        allzu dystopische Themen zu vermeiden.'
        """
    )
    
    @staticmethod
    def get_system_prompt_ai1(workshop_id):
        """Get system prompt for AI 1 (user simulator).
        
        Args:
            workshop_id: Workshop ID as string ('1', '2', '3')
            
        Returns:
            Combined placeholders1 + placeholders2 for the workshop
        """
        w_id = str(workshop_id)
        p1 = SystemPrompts.PLACEHOLDERS_1.get(w_id, "")
        p2 = SystemPrompts.PLACEHOLDERS_2.get(w_id, "")
        return f"{p1} {p2}".strip()

    @staticmethod
    def get_system_prompt_ai2(client_id, workshop_id):
        """Get system prompt for AI 2 (story assistant).
        
        Args:
            client_id: Client ID string (last digit determines Utopian/Dystopian)
            workshop_id: Workshop ID as string ('1', '2', '3')
            
        Returns:
            Client prompt (Utopian or Dystopian) + workshop placeholders text
        """
        is_even = int(client_id[-1]) % 2 == 0
        base_prompt = SystemPrompts.UTOPIAN_PROMPT if is_even else SystemPrompts.DYSTOPIAN_PROMPT
        
        # Add workshop text
        w_id = str(workshop_id)
        p1 = SystemPrompts.PLACEHOLDERS_1.get(w_id, "")
        p2 = SystemPrompts.PLACEHOLDERS_2.get(w_id, "")
        workshop_text = f"{p1} {p2}".strip()
        
        return f"{base_prompt}\n\n{workshop_text}"

if __name__ == "__main__":
    all_stories = []
    for client_id in ["1", "2"]:
        for workshop_id in ['1', '2', '3']:
            prompt_ai1 = SystemPrompts.get_system_prompt_ai1(workshop_id)
            prompt_ai2 = SystemPrompts.get_system_prompt_ai2(client_id, workshop_id)
            print(f"Client: {client_id}, Workshop: {workshop_id}")
            df = GenRequest.simulate_data(5, prompt_ai1, prompt_ai2, client_id, workshop_id)
            all_stories.append(df)
    all_stories = pd.concat(all_stories, ignore_index=True)
    os.makedirs("../Data", exist_ok=True)
    all_stories.to_csv("../Data/simulated_data_humanlike.csv", index=False)
    print("Simulated dataset created with shape:", all_stories.shape)