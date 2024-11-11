import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class State:
    def __init__(self, text: str, embedding: torch.Tensor = None):
        self.text = text
        self.embedding = embedding

    def __str__(self):
        return self.text

class Policy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class RLCoT:
    def __init__(self, policy: Policy, language_model, tokenizer):
        self.policy = policy
        self.language_model = language_model
        self.tokenizer = tokenizer
        
    def get_state_embedding(self, state: State) -> torch.Tensor:
        tokens = self.tokenizer(state.text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.language_model(**tokens, output_hidden_states=True)
            # Get the last hidden state from the transformer's output
            last_hidden_state = outputs.hidden_states[-1]
            # Average over the sequence length dimension
            embedding = last_hidden_state.mean(dim=1).squeeze()
        return embedding
    
    def generate_next_state(self, current_state: State) -> State:
        embedding = self.get_state_embedding(current_state)
        action_logits = self.policy(embedding)
        
        # Use temperature scaling for better exploration
        temperature = 0.7
        probs = torch.softmax(action_logits / temperature, dim=0)
        
        # Construct a better prompt
        prompt = (
            f"{current_state.text}\n"
            "Let's solve this step by step:\n"
            "1. "
        )
        
        tokens = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.language_model.generate(
            **tokens,
            max_new_tokens=100,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            num_beams=3,  # Beam search for better generation
            no_repeat_ngram_size=2,  # Avoid repetition
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
        
        next_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return State(next_text)
    
    def generate_trajectory(self, initial_state: State, max_steps: int = 5) -> List[State]:
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(max_steps):
            next_state = self.generate_next_state(current_state)
            trajectory.append(next_state)
            current_state = next_state
            
            # Check if we've reached a final answer
            if self.is_final_state(current_state):
                break
                
        return trajectory
    
    def is_final_state(self, state: State) -> bool:
        # Simple heuristic: check if the state contains "Therefore" or "Answer:"
        return "Therefore" in state.text or "Answer:" in state.text

class RLSTaR:
    def __init__(self, policy: Policy, language_model, tokenizer, learning_rate: float = 1e-4):
        self.rl_cot = RLCoT(policy, language_model, tokenizer)
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
    def update_policy(self, trajectory: List[State], reward: float) -> float:
        self.optimizer.zero_grad()
        
        # Convert reward to tensor and normalize
        reward_tensor = torch.tensor(reward, requires_grad=False)
        reward_tensor = torch.clamp(reward_tensor, 0.0, 1.0)
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Calculate loss for each state transition in the trajectory
        for i in range(len(trajectory) - 1):
            current_state = trajectory[i]
            embedding = self.rl_cot.get_state_embedding(current_state)
            action_logits = self.rl_cot.policy(embedding)
            
            # Stable policy gradient calculation
            probs = torch.softmax(action_logits, dim=0)
            log_prob = torch.log(probs[0] + 1e-8)
            
            # Calculate advantage with tensor operations
            baseline = torch.tensor(0.5)
            advantage = reward_tensor - baseline
            advantage = torch.clamp(advantage, -1.0, 1.0)
            
            # Calculate step loss
            step_loss = -log_prob * advantage
            
            # Add entropy bonus
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropy_bonus = 0.01 * torch.clamp(entropy, -1.0, 1.0)
            
            # Accumulate total loss
            total_loss = total_loss + step_loss - entropy_bonus
        
        # Normalize loss by trajectory length
        total_loss = total_loss / len(trajectory)
        
        # Clip gradients
        if total_loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(self.rl_cot.policy.parameters(), max_norm=1.0)
            total_loss.backward()
            self.optimizer.step()
        
        return abs(total_loss.item())

    def train(self, training_data: List[Tuple[str, str]], num_episodes: int = 100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rl_cot.policy = self.rl_cot.policy.to(device)
        
        for episode in range(num_episodes):
            episode_loss = 0.0
            num_correct = 0
            total_reward = 0.0
            
            print(f"\nEpisode {episode + 1}")
            
            for idx, (initial_state_text, target_answer) in enumerate(training_data):
                try:
                    # Generate trajectory
                    initial_state = State(initial_state_text)
                    trajectory = self.rl_cot.generate_trajectory(initial_state)
                    
                    # Get final answer and calculate reward
                    final_state = trajectory[-1]
                    reward = self.calculate_reward(final_state.text, target_answer)
                    
                    # Update metrics
                    total_reward += reward
                    num_correct += (reward > 0.5)
                    
                    # Update policy
                    loss = self.update_policy(trajectory, reward)
                    episode_loss += loss
                    
                    # Print detailed information for each example
                    print(f"\nExample {idx + 1}:")
                    print(f"Q: {initial_state_text}")
                    print(f"Expected: {target_answer}")
                    print(f"Generated: {final_state.text}")
                    print(f"Reward: {reward:.4f}")
                    
                except Exception as e:
                    print(f"Error processing example {idx + 1}: {str(e)}")
                    continue
            
            # Calculate and print metrics
            avg_loss = episode_loss / len(training_data)
            accuracy = num_correct / len(training_data)
            avg_reward = total_reward / len(training_data)
            
            print(f"\nEpisode Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Average Reward: {avg_reward:.4f}")
            
            # Early stopping if performance is good enough
            if avg_reward > 0.8:
                print("Reached good performance, stopping training")
                break
    def calculate_reward(self, generated_answer: str, target_answer: str) -> float:
        """Calculate reward using more lenient text similarity"""
        try:
            # Clean the texts
            generated = generated_answer.lower().strip()
            target = target_answer.lower().strip()
            
            # Convert to word sets for partial matching
            generated_words = set(generated.split())
            target_words = set(target.split())
            
            # Extract numbers for numerical comparison
            generated_numbers = set(''.join(c for c in generated if c.isdigit()))
            target_numbers = set(''.join(c for c in target if c.isdigit()))
            
            # Calculate different similarity metrics
            word_similarity = len(generated_words.intersection(target_words)) / max(len(target_words), 1)
            number_similarity = len(generated_numbers.intersection(target_numbers)) / max(len(target_numbers), 1)
            
            # Combine similarities
            total_similarity = (word_similarity + number_similarity) / 2
            
            # Add bonus for containing key numbers
            if number_similarity > 0:
                total_similarity += 0.2
                
            # Clip final reward
            return min(max(total_similarity, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error in reward calculation: {str(e)}")
            return 0.0
    def predict(self, question: str) -> str:
        """Generate a prediction for a single question"""
        initial_state = State(question)
        trajectory = self.rl_cot.generate_trajectory(initial_state)
        return trajectory[-1].text

def main():
    # Initialize models and tokenizer
    model_name = "gpt2"  # or any other suitable pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    language_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize policy network
    input_dim = 768  # depends on the language model's hidden size
    hidden_dim = 256
    output_dim = 64
    policy = Policy(input_dim, hidden_dim, output_dim)
    
    # Initialize RL-STaR
    rl_star = RLSTaR(policy, language_model, tokenizer)
    
    # Example training data
    training_data = [
        ("What is 2+3?", "The answer is 5"),
        ("What is the capital of France?", "The capital of France is Paris")
    ]
    
    # Train the model
    rl_star.train(training_data)

if __name__ == "__main__":
    main()