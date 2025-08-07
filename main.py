import networkx as nx
import numpy as np
import json
import threading
import time
import asyncio
from datetime import datetime, timedelta
from collections import deque, defaultdict
import sqlite3
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the correct Google GenAI API
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    logger.error("❌ google-genai package not installed. Please install: pip install google-genai")
    GENAI_AVAILABLE = False

class LLMEnhancedConceptualMind:
    def __init__(self, brain_name="GeminiDeepMind", api_key=None, persist_data=True):
        self.brain_name = brain_name
        self.persist_data = persist_data
        self.data_file = f"{brain_name}_knowledge.db"
        
        # Main concept graph
        self.concept_graph = nx.Graph()
        
        # Enhanced memory systems
        self.short_term_memory = deque(maxlen=1000)
        self.long_term_memory = {}
        self.working_memory = {}
        self.concept_strengths = defaultdict(float)
        self.relationship_history = defaultdict(list)
        self.llm_query_history = []
        self.learning_patterns = defaultdict(list)
        
        # Advanced learning parameters
        self.curiosity_score = 0.8
        self.learning_rate = 0.15
        self.connection_threshold = 0.6
        self.max_connections_per_concept = 10
        self.concept_decay_rate = 0.95
        self.reinforcement_factor = 1.1
        
        # Gemini interaction settings
        self.max_tokens = 2000
        self.api_calls_count = 0
        self.max_api_calls_per_session = 100
        self.gemini_api_key = "AIzaSyCQd5Ere8o3dy_VFRp19j78vjjvh3e1YTk"  # Replace with your Gemini API key
        self.model_name = "gemini-2.5-flash"  # Free-tier model
        
        # Learning control
        self.is_learning = False
        self.learning_thread = None
        self.lock = threading.Lock()
        self.session_start_time = datetime.now()
        self.last_save_time = datetime.now()
        
        # Learning modes with probabilities
        self.learning_modes = {
            'exploration': 0.4,
            'connection': 0.3,
            'deepening': 0.2,
            'consolidation': 0.1
        }
        
        # Initialize systems
        self.client = None
        self._setup_gemini_client()
        self._load_persistent_data()
        self.initialize_seed_concepts()
        
        logger.info(f"🧠 {self.brain_name} initialized with continuous learning!")
        logger.info(f"🔮 Session limit: {self.max_api_calls_per_session} API calls")
    
    def _setup_gemini_client(self):
        """Setup Gemini client with proper error handling"""
        if not GENAI_AVAILABLE:
            logger.error("❌ Google GenAI package not available")
            return
            
        try:
            if self.gemini_api_key == "YOUR_API_KEY_HERE" or not self.gemini_api_key:
                logger.warning("⚠️ Please set your Gemini API key!")
                return
            
            # Create client with API key
            self.client = genai.Client(api_key=self.gemini_api_key)
            logger.info("✅ Gemini client configured successfully")
            
            # Test the client with a simple call
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents="Hello, respond with 'OK' if you can hear me."
                )
                if response and response.text:
                    logger.info("✅ Gemini API connection verified")
                else:
                    logger.warning("⚠️ Gemini API test failed - no response")
                    self.client = None
            except Exception as test_error:
                logger.error(f"❌ Gemini API test failed: {test_error}")
                self.client = None
                
        except Exception as e:
            logger.error(f"❌ Failed to setup Gemini client: {e}")
            self.client = None
    
    def _load_persistent_data(self):
        """Load persistent knowledge from database"""
        if not self.persist_data:
            return
            
        try:
            if os.path.exists(self.data_file):
                conn = sqlite3.connect(self.data_file)
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'")
                if not cursor.fetchone():
                    conn.close()
                    return
                
                # Load concepts
                cursor.execute("SELECT * FROM concepts")
                for row in cursor.fetchall():
                    try:
                        concept_data = json.loads(row[1])
                        self.concept_graph.add_node(row[0], **concept_data)
                    except json.JSONDecodeError:
                        continue
                
                # Load relationships
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'")
                if cursor.fetchone():
                    cursor.execute("SELECT * FROM relationships")
                    for row in cursor.fetchall():
                        try:
                            edge_data = json.loads(row[2])
                            self.concept_graph.add_edge(row[0], row[1], **edge_data)
                        except json.JSONDecodeError:
                            continue
                
                # Load metadata
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
                if cursor.fetchone():
                    cursor.execute("SELECT * FROM metadata WHERE key = 'concept_strengths'")
                    result = cursor.fetchone()
                    if result:
                        try:
                            self.concept_strengths = defaultdict(float, json.loads(result[1]))
                        except json.JSONDecodeError:
                            pass
                
                conn.close()
                logger.info(f"📚 Loaded {len(self.concept_graph.nodes())} concepts from persistent storage")
        except Exception as e:
            logger.error(f"❌ Error loading persistent data: {e}")
    
    def _save_persistent_data(self):
        """Save knowledge to database"""
        if not self.persist_data:
            return
            
        try:
            conn = sqlite3.connect(self.data_file)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    name TEXT PRIMARY KEY,
                    data TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    concept1 TEXT,
                    concept2 TEXT,
                    data TEXT,
                    PRIMARY KEY (concept1, concept2)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Save concepts
            cursor.execute("DELETE FROM concepts")
            for node, data in self.concept_graph.nodes(data=True):
                try:
                    cursor.execute("INSERT INTO concepts (name, data) VALUES (?, ?)",
                                 (node, json.dumps(data, default=str)))
                except Exception:
                    continue
            
            # Save relationships
            cursor.execute("DELETE FROM relationships")
            for edge in self.concept_graph.edges(data=True):
                try:
                    cursor.execute("INSERT INTO relationships (concept1, concept2, data) VALUES (?, ?, ?)",
                                 (edge[0], edge[1], json.dumps(edge[2], default=str)))
                except Exception:
                    continue
            
            # Save metadata
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                          ("concept_strengths", json.dumps(dict(self.concept_strengths))))
            
            conn.commit()
            conn.close()
            self.last_save_time = datetime.now()
            logger.info("💾 Knowledge saved to persistent storage")
        except Exception as e:
            logger.error(f"❌ Error saving persistent data: {e}")
    
    def initialize_seed_concepts(self):
        """Initialize with enhanced seed concepts"""
        if len(self.concept_graph.nodes()) > 0:
            return  # Already loaded from persistent storage
            
        seed_concepts = {
            "consciousness": {"category": "philosophical", "depth": 1.0, "mystery": 1.0, "importance": 0.9},
            "light": {"category": "physical", "depth": 0.8, "mystery": 0.7, "importance": 0.7},
            "pattern": {"category": "abstract", "depth": 0.7, "mystery": 0.6, "importance": 0.6},
            "energy": {"category": "physical", "depth": 0.8, "mystery": 0.7, "importance": 0.8},
            "time": {"category": "philosophical", "depth": 0.95, "mystery": 0.95, "importance": 0.9},
            "learning": {"category": "cognitive", "depth": 0.85, "mystery": 0.8, "importance": 0.8},
            "connection": {"category": "abstract", "depth": 0.75, "mystery": 0.7, "importance": 0.7},
            "emergence": {"category": "systems", "depth": 0.9, "mystery": 0.9, "importance": 0.8}
        }
        
        for concept, properties in seed_concepts.items():
            properties.update({
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0
            })
            self.concept_graph.add_node(concept, **properties)
            self.concept_strengths[concept] = properties.get('depth', 0.5)
        
        logger.info(f"🌱 Initialized with {len(seed_concepts)} seed concepts")
    
    def query_llm_sync(self, prompt):
        """Synchronous Gemini query using the correct API"""
        if not self.client:
            return None
            
        if self.api_calls_count >= self.max_api_calls_per_session:
            logger.warning(f"⚠️ API call limit reached ({self.max_api_calls_per_session})")
            return None
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            self.api_calls_count += 1
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("⚠️ Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}")
            return None
    
    async def query_llm_async(self, prompt):
        """Asynchronous Gemini query using the correct API"""
        if not self.client:
            return None
            
        if self.api_calls_count >= self.max_api_calls_per_session:
            logger.warning(f"⚠️ API call limit reached ({self.max_api_calls_per_session})")
            return None
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            self.api_calls_count += 1
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("⚠️ Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}")
            return None
    
    def discover_concept_with_llm(self, concept):
        """Enhanced concept discovery with better prompting"""
        logger.info(f"🔍 Discovering concept: '{concept}'")
        
        # Check if concept already exists
        if concept in self.concept_graph:
            self._update_concept_access(concept)
            return self.concept_graph.nodes[concept].get('definition', 'No definition available')
        
        # Create sophisticated prompt
        existing_concepts = list(self.concept_graph.nodes())[:10]
        
        prompt = f"""
        Analyze the concept "{concept}" deeply and provide insights in the following format:
        
        Definition: [Clear, concise definition]
        Category: [philosophical/physical/abstract/cognitive/systems/social/biological]
        Depth Score: [0.0 to 1.0 - how fundamental/complex this concept is]
        Mystery Score: [0.0 to 1.0 - how much unknown/unexplored aspects remain]
        Importance Score: [0.0 to 1.0 - significance in understanding reality]
        
        Key Properties: [List 3-5 essential characteristics]
        
        Relationships: [How this connects to: {', '.join(existing_concepts[:5])}]
        
        Deep Insight: [Profound understanding or implications]
        
        Questions Raised: [2-3 deeper questions this concept opens up]
        """
        
        analysis = self.query_llm_sync(prompt)
        
        if analysis:
            # Parse the response and create concept
            concept_data = self._parse_llm_concept_response(analysis, concept)
            
            self.concept_graph.add_node(concept, **concept_data)
            self.concept_strengths[concept] = concept_data.get('depth', 0.5)
            
            # Add to memory
            self.short_term_memory.append({
                'type': 'concept_discovery',
                'concept': concept,
                'timestamp': datetime.now(),
                'source': 'llm',
                'data': concept_data
            })
            
            # Try to establish connections
            self._establish_concept_connections(concept, analysis)
            
            logger.info(f"✨ Discovered concept '{concept}' with {concept_data.get('category', 'unknown')} category")
            return analysis
        
        return self._fallback_concept_creation(concept)
    
    def _parse_llm_concept_response(self, response, concept):
        """Parse LLM response into structured concept data"""
        data = {
            'name': concept,
            'definition': '',
            'category': 'unknown',
            'depth': 0.5,
            'mystery': 0.5,
            'importance': 0.5,
            'properties': [],
            'deep_insight': '',
            'questions_raised': [],
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1,
            'source': 'llm'
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Definition:'):
                data['definition'] = line.replace('Definition:', '').strip()
            elif line.startswith('Category:'):
                category = line.replace('Category:', '').strip().lower()
                if category in ['philosophical', 'physical', 'abstract', 'cognitive', 'systems', 'social', 'biological']:
                    data['category'] = category
            elif line.startswith('Depth Score:'):
                try:
                    score = float(line.split(':')[1].strip().split()[0])
                    data['depth'] = max(0.0, min(1.0, score))
                except:
                    pass
            elif line.startswith('Mystery Score:'):
                try:
                    score = float(line.split(':')[1].strip().split()[0])
                    data['mystery'] = max(0.0, min(1.0, score))
                except:
                    pass
            elif line.startswith('Importance Score:'):
                try:
                    score = float(line.split(':')[1].strip().split()[0])
                    data['importance'] = max(0.0, min(1.0, score))
                except:
                    pass
            elif line.startswith('Deep Insight:'):
                data['deep_insight'] = line.replace('Deep Insight:', '').strip()
        
        return data
    
    def _establish_concept_connections(self, new_concept, analysis):
        """Establish connections between concepts based on analysis"""
        existing_concepts = list(self.concept_graph.nodes())
        
        for existing_concept in existing_concepts:
            if existing_concept == new_concept:
                continue
                
            # Simple connection heuristic based on text analysis
            if existing_concept.lower() in analysis.lower():
                connection_strength = np.random.uniform(0.3, 0.8)
                
                if connection_strength > self.connection_threshold:
                    self.concept_graph.add_edge(
                        new_concept, existing_concept,
                        strength=connection_strength,
                        type='semantic',
                        created_at=datetime.now().isoformat(),
                        source='discovery'
                    )
                    logger.info(f"🔗 Connected '{new_concept}' to '{existing_concept}' (strength: {connection_strength:.2f})")
    
    def _update_concept_access(self, concept):
        """Update concept access statistics"""
        if concept in self.concept_graph:
            self.concept_graph.nodes[concept]['last_accessed'] = datetime.now().isoformat()
            self.concept_graph.nodes[concept]['access_count'] += 1
            self.concept_strengths[concept] *= self.reinforcement_factor
    
    def _fallback_concept_creation(self, concept):
        """Create concept when LLM is unavailable"""
        data = {
            'name': concept,
            'definition': f"Concept requiring exploration: {concept}",
            'category': 'unknown',
            'depth': 0.5,
            'mystery': 0.6,
            'importance': 0.5,
            'created_at': datetime.now().isoformat(),
            'source': 'fallback'
        }
        
        self.concept_graph.add_node(concept, **data)
        self.concept_strengths[concept] = 0.5
        
        return data['definition']
    
    async def explore_concept_relationship_async(self, concept1, concept2):
        """Async exploration of concept relationships"""
        info1 = self.concept_graph.nodes.get(concept1, {})
        info2 = self.concept_graph.nodes.get(concept2, {})
        
        prompt = f"""
        Explore the deep relationship between "{concept1}" and "{concept2}".
        
        {concept1}: {info1.get('definition', 'Unknown concept')}
        {concept2}: {info2.get('definition', 'Unknown concept')}
        
        Please analyze:
        1. Direct connections between these concepts
        2. Indirect or emergent relationships
        3. How they might influence each other
        4. What new insights arise from their interaction
        5. Strength of relationship (0.0 to 1.0)
        6. Type of relationship (causal/semantic/structural/emergent/oppositional)
        
        Provide deep philosophical and scientific perspectives.
        """
        
        analysis = await self.query_llm_async(prompt)
        
        if analysis:
            # Extract relationship strength and type
            strength = self._extract_relationship_strength(analysis)
            rel_type = self._extract_relationship_type(analysis)
            
            if not self.concept_graph.has_edge(concept1, concept2) and strength > self.connection_threshold:
                self.concept_graph.add_edge(
                    concept1, concept2,
                    strength=strength,
                    type=rel_type,
                    analysis=analysis,
                    created_at=datetime.now().isoformat(),
                    source='llm_exploration'
                )
                
                logger.info(f"🔗 New relationship: {concept1} --[{rel_type}:{strength:.2f}]--> {concept2}")
                
                # Add to relationship history
                self.relationship_history[f"{concept1}-{concept2}"].append({
                    'timestamp': datetime.now(),
                    'strength': strength,
                    'type': rel_type,
                    'analysis': analysis[:200] + "..."
                })
        
        return analysis
    
    def _extract_relationship_strength(self, analysis):
        """Extract relationship strength from analysis"""
        try:
            # Look for strength indicators in the text
            lines = analysis.lower().split('\n')
            for line in lines:
                if 'strength' in line and any(char.isdigit() for char in line):
                    # Extract number between 0 and 1
                    import re
                    numbers = re.findall(r'0?\.\d+|[01]\.?\d*', line)
                    for num in numbers:
                        val = float(num)
                        if 0 <= val <= 1:
                            return val
        except:
            pass
        
        # Default strength based on content analysis
        connection_words = ['strong', 'deep', 'fundamental', 'essential', 'critical']
        weak_words = ['weak', 'minor', 'superficial', 'limited']
        
        strength = 0.5
        for word in connection_words:
            if word in analysis.lower():
                strength += 0.1
        for word in weak_words:
            if word in analysis.lower():
                strength -= 0.1
        
        return max(0.1, min(0.9, strength))
    
    def _extract_relationship_type(self, analysis):
        """Extract relationship type from analysis"""
        analysis_lower = analysis.lower()
        
        if 'causal' in analysis_lower or 'cause' in analysis_lower:
            return 'causal'
        elif 'semantic' in analysis_lower or 'meaning' in analysis_lower:
            return 'semantic'
        elif 'structural' in analysis_lower or 'structure' in analysis_lower:
            return 'structural'
        elif 'emergent' in analysis_lower or 'emerge' in analysis_lower:
            return 'emergent'
        elif 'opposite' in analysis_lower or 'contrast' in analysis_lower:
            return 'oppositional'
        else:
            return 'associative'
    
    def continuous_learning_cycle(self):
        """Main continuous learning cycle"""
        cycle_count = 0
        
        while self.is_learning and self.api_calls_count < self.max_api_calls_per_session:
            try:
                cycle_count += 1
                logger.info(f"\n🔄 Learning cycle {cycle_count} (API: {self.api_calls_count}/{self.max_api_calls_per_session})")
                
                # Choose learning mode based on probabilities
                mode_names = list(self.learning_modes.keys())
                mode_probs = list(self.learning_modes.values())
                
                # Fix the numpy random choice issue
                if len(mode_names) > 1:
                    mode = np.random.choice(mode_names, p=mode_probs)
                else:
                    mode = mode_names[0]
                
                if mode == 'exploration':
                    self._exploration_mode()
                elif mode == 'connection':
                    # Run async function in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._connection_mode())
                    finally:
                        loop.close()
                elif mode == 'deepening':
                    self._deepening_mode()
                elif mode == 'consolidation':
                    self._consolidation_mode()
                
                # Periodic maintenance
                if cycle_count % 10 == 0:
                    self._decay_unused_concepts()
                    self._save_persistent_data()
                
                # Adaptive sleep based on activity
                sleep_time = max(3, 8 - (self.api_calls_count / 10))
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in learning cycle: {e}")
                time.sleep(5)
        
        logger.info("⏹️ Continuous learning cycle completed")
        self.is_learning = False
    
    def _exploration_mode(self):
        """Explore new concepts based on current knowledge"""
        logger.info("🔍 Exploration mode activated")
        
        # Generate new concept to explore
        existing_concepts = list(self.concept_graph.nodes())
        if not existing_concepts:
            return
            
        seed_concept = np.random.choice(existing_concepts)
        
        prompt = f"""
        Based on the concept "{seed_concept}", suggest 3 related but unexplored concepts 
        that would deepen understanding. Focus on:
        - Fundamental principles
        - Emergent properties
        - Cross-disciplinary connections
        - Philosophical implications
        
        Just list the concept names, one per line.
        """
        
        response = self.query_llm_sync(prompt)
        if response:
            new_concepts = [line.strip().lower() for line in response.split('\n') 
                          if line.strip() and line.strip().lower() not in existing_concepts]
            
            if new_concepts:
                chosen_concept = new_concepts[0]
                self.discover_concept_with_llm(chosen_concept)
    
    async def _connection_mode(self):
        """Focus on finding connections between existing concepts"""
        logger.info("🔗 Connection mode activated")
        
        concepts = list(self.concept_graph.nodes())
        if len(concepts) < 2:
            return
        
        # Find concepts with few connections
        connection_counts = dict(self.concept_graph.degree())
        underconnected = [c for c, degree in connection_counts.items() 
                         if degree < self.max_connections_per_concept // 2]
        
        if underconnected:
            concept1 = np.random.choice(underconnected)
            remaining_concepts = [c for c in concepts if c != concept1]
            if remaining_concepts:
                concept2 = np.random.choice(remaining_concepts)
            else:
                return
        else:
            if len(concepts) >= 2:
                selected = np.random.choice(concepts, 2, replace=False)
                concept1, concept2 = selected[0], selected[1]
            else:
                return
        
        if not self.concept_graph.has_edge(concept1, concept2):
            await self.explore_concept_relationship_async(concept1, concept2)
    
    def _deepening_mode(self):
        """Deepen understanding of existing concepts"""
        logger.info("🏔️ Deepening mode activated")
        
        concepts = list(self.concept_graph.nodes(data=True))
        if not concepts:
            return
            
        # Focus on important but shallow concepts
        shallow_important = [(name, data) for name, data in concepts 
                           if data.get('importance', 0.5) > 0.7 and data.get('depth', 0.5) < 0.8]
        
        if shallow_important:
            concept, data = np.random.choice(len(shallow_important))
            concept, data = shallow_important[concept]
        else:
            concept, data = concepts[np.random.choice(len(concepts))]
        
        prompt = f"""
        Deepen the understanding of "{concept}".
        Current definition: {data.get('definition', 'No definition')}
        
        Provide:
        1. Advanced insights and implications
        2. Historical development of this concept
        3. Current frontiers and unknowns
        4. Practical applications
        5. Philosophical significance
        
        Focus on what makes this concept profound and mysterious.
        """
        
        response = self.query_llm_sync(prompt)
        if response:
            # Update concept with deeper insights
            self.concept_graph.nodes[concept].update({
                'deep_analysis': response,
                'last_deepened': datetime.now().isoformat(),
                'depth': min(1.0, data.get('depth', 0.5) + 0.1)
            })
            
            logger.info(f"🏔️ Deepened understanding of '{concept}'")
    
    def _consolidation_mode(self):
        """Consolidate and strengthen existing knowledge"""
        logger.info("💪 Consolidation mode activated")
        
        # Strengthen frequently accessed concepts
        for concept in self.concept_graph.nodes():
            access_count = self.concept_graph.nodes[concept].get('access_count', 0)
            if access_count > 5:
                self.concept_strengths[concept] *= 1.05
        
        # Remove very weak connections
        weak_edges = [(u, v) for u, v, d in self.concept_graph.edges(data=True)
                     if d.get('strength', 0.5) < 0.3]
        
        if weak_edges and len(weak_edges) < len(self.concept_graph.edges()) * 0.1:
            edge_to_remove = weak_edges[np.random.choice(len(weak_edges))]
            u, v = edge_to_remove
            self.concept_graph.remove_edge(u, v)
            logger.info(f"🗑️ Removed weak connection: {u} -- {v}")
    
    def _decay_unused_concepts(self):
        """Apply decay to unused concepts"""
        current_time = datetime.now()
        
        for concept in list(self.concept_graph.nodes()):
            last_accessed = self.concept_graph.nodes[concept].get('last_accessed')
            if last_accessed:
                try:
                    last_time = datetime.fromisoformat(last_accessed)
                    hours_since_access = (current_time - last_time).total_seconds() / 3600
                    
                    if hours_since_access > 24:  # Decay after 24 hours
                        self.concept_strengths[concept] *= self.concept_decay_rate
                        
                        # Remove very weak concepts
                        if self.concept_strengths[concept] < 0.1:
                            self.concept_graph.remove_node(concept)
                            del self.concept_strengths[concept]
                            logger.info(f"🗑️ Removed weak concept: {concept}")
                except ValueError:
                    # Handle invalid datetime format
                    continue
    
    def start_continuous_learning(self):
        """Start the continuous learning system"""
        if self.is_learning:
            logger.warning("⚠️ Learning already in progress")
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(
            target=self.continuous_learning_cycle, 
            daemon=True
        )
        self.learning_thread.start()
        logger.info("🚀 Continuous learning started!")
    
    def stop_learning(self):
        """Stop the learning system"""
        self.is_learning = False
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=10)
        self._save_persistent_data()
        logger.info("⏹️ Learning stopped and knowledge saved")
    
    def query_knowledge(self, query):
        """Query the knowledge base"""
        logger.info(f"❓ Querying: '{query}'")
        
        # Find relevant concepts
        relevant_concepts = []
        query_lower = query.lower()
        
        for concept in self.concept_graph.nodes():
            if query_lower in concept.lower() or concept.lower() in query_lower:
                relevant_concepts.append(concept)
        
        if not relevant_concepts:
            # Use LLM to find concepts
            concept_search_prompt = f"""
            Given this query: "{query}"
            From these available concepts: {list(self.concept_graph.nodes())[:20]}
            Which concepts are most relevant? List up to 5.
            """
            response = self.query_llm_sync(concept_search_prompt)
            if response:
                relevant_concepts = [c.strip() for c in response.split('\n') if c.strip()]
        
        # Get information about relevant concepts
        result = {"query": query, "relevant_concepts": [], "insights": []}
        
        for concept in relevant_concepts[:5]:
            if concept in self.concept_graph:
                concept_data = self.concept_graph.nodes[concept]
                result["relevant_concepts"].append({
                    "name": concept,
                    "definition": concept_data.get('definition', 'No definition'),
                    "depth": concept_data.get('depth', 0.5),
                    "connections": list(self.concept_graph.neighbors(concept))
                })
                self._update_concept_access(concept)
        
        # Generate insights using LLM
        if relevant_concepts and self.client:
            insight_prompt = f"""
            Query: "{query}"
            Relevant concepts: {relevant_concepts}
            
            Based on the knowledge about these concepts, provide deep insights 
            that answer the query. Connect the concepts meaningfully.
            """
            insights = self.query_llm_sync(insight_prompt)
            if insights:
                result["insights"] = [insights]
        
        return result
    
    def get_knowledge_summary(self):
        """Get comprehensive knowledge summary"""
        total_concepts = len(self.concept_graph.nodes())
        total_connections = len(self.concept_graph.edges())
        
        # Categories
        categories = defaultdict(int)
        depths = []
        mysteries = []
        
        for concept, data in self.concept_graph.nodes(data=True):
            categories[data.get('category', 'unknown')] += 1
            depths.append(data.get('depth', 0.5))
            mysteries.append(data.get('mystery', 0.5))
        
        # Most connected concepts
        degree_centrality = nx.degree_centrality(self.concept_graph)
        top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Strongest concepts
        top_strong = sorted(self.concept_strengths.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Learning statistics
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        
        return {
            'brain_name': self.brain_name,
            'session_duration_minutes': round(session_duration, 1),
            'total_concepts': total_concepts,
            'total_connections': total_connections,
            'api_calls_used': self.api_calls_count,
            'api_calls_remaining': self.max_api_calls_per_session - self.api_calls_count,
            'categories': dict(categories),
            'average_depth': round(np.mean(depths), 3) if depths else 0,
            'average_mystery': round(np.mean(mysteries), 3) if mysteries else 0,
            'top_connected': top_connected,
            'strongest_concepts': top_strong,
            'is_learning': self.is_learning,
            'learning_modes': self.learning_modes,
            'recent_discoveries': list(self.short_term_memory)[-5:] if self.short_term_memory else []
        }
    
    def visualize_knowledge(self, detailed=False):
        """Visualize the current knowledge state"""
        summary = self.get_knowledge_summary()
        
        print(f"""
🌌 === {summary['brain_name']} Knowledge State ===
⏱️  Session Duration: {summary['session_duration_minutes']} minutes
📊 Concepts: {summary['total_concepts']} | Connections: {summary['total_connections']}
🤖 API Usage: {summary['api_calls_used']}/{summary['api_calls_used'] + summary['api_calls_remaining']} calls
🧠 Learning: {'🟢 ACTIVE' if summary['is_learning'] else '🔴 INACTIVE'}

📈 Knowledge Metrics:
   • Average Depth: {summary['average_depth']}
   • Average Mystery: {summary['average_mystery']}

🏷️  Categories: {dict(summary['categories'])}

🔗 Most Connected Concepts:""")
        
        for concept, centrality in summary['top_connected']:
            connections = len(list(self.concept_graph.neighbors(concept)))
            print(f"   • {concept}: {connections} connections")
        
        print(f"\n💪 Strongest Concepts:")
        for concept, strength in summary['strongest_concepts']:
            print(f"   • {concept}: {strength:.3f}")
        
        if detailed and summary['recent_discoveries']:
            print(f"\n🔍 Recent Discoveries:")
            for discovery in summary['recent_discoveries']:
                if discovery['type'] == 'concept_discovery':
                    print(f"   • {discovery['concept']} ({discovery.get('timestamp', 'unknown time')})")
        
        print(f"\n🎯 Learning Mode Distribution:")
        for mode, probability in summary['learning_modes'].items():
            print(f"   • {mode.capitalize()}: {probability:.1%}")
    
    def add_user_concept(self, concept_name, user_definition=None):
        """Allow users to add concepts manually"""
        if concept_name in self.concept_graph:
            logger.info(f"⚠️ Concept '{concept_name}' already exists")
            return self.concept_graph.nodes[concept_name]
        
        if user_definition:
            # Use user definition
            concept_data = {
                'name': concept_name,
                'definition': user_definition,
                'category': 'user_defined',
                'depth': 0.5,
                'mystery': 0.5,
                'importance': 0.6,
                'created_at': datetime.now().isoformat(),
                'source': 'user'
            }
            
            self.concept_graph.add_node(concept_name, **concept_data)
            self.concept_strengths[concept_name] = 0.5
            
            logger.info(f"✅ User concept '{concept_name}' added")
            return concept_data
        else:
            # Use LLM to analyze user's concept
            return self.discover_concept_with_llm(concept_name)
    
    def explore_concept_cluster(self, center_concept, depth=2):
        """Explore a cluster of related concepts"""
        if center_concept not in self.concept_graph:
            logger.warning(f"⚠️ Concept '{center_concept}' not found")
            return None
        
        # Get concepts within specified depth
        cluster = set([center_concept])
        current_level = set([center_concept])
        
        for _ in range(depth):
            next_level = set()
            for concept in current_level:
                neighbors = set(self.concept_graph.neighbors(concept))
                next_level.update(neighbors)
                cluster.update(neighbors)
            current_level = next_level - cluster
            if not current_level:
                break
        
        # Create subgraph
        subgraph = self.concept_graph.subgraph(cluster)
        
        # Analyze cluster
        cluster_info = {
            'center_concept': center_concept,
            'cluster_size': len(cluster),
            'concepts': list(cluster),
            'connections': len(subgraph.edges()),
            'density': nx.density(subgraph) if len(cluster) > 1 else 0,
            'categories': defaultdict(int)
        }
        
        for concept in cluster:
            category = self.concept_graph.nodes[concept].get('category', 'unknown')
            cluster_info['categories'][category] += 1
        
        logger.info(f"🕸️ Explored cluster around '{center_concept}': {len(cluster)} concepts")
        return cluster_info
    
    def get_concept_path(self, concept1, concept2):
        """Find path between two concepts"""
        if concept1 not in self.concept_graph or concept2 not in self.concept_graph:
            return None
        
        try:
            path = nx.shortest_path(self.concept_graph, concept1, concept2)
            path_info = {
                'path': path,
                'length': len(path) - 1,
                'connections': []
            }
            
            for i in range(len(path) - 1):
                edge_data = self.concept_graph.edges[path[i], path[i+1]]
                path_info['connections'].append({
                    'from': path[i],
                    'to': path[i+1],
                    'strength': edge_data.get('strength', 0.5),
                    'type': edge_data.get('type', 'unknown')
                })
            
            return path_info
        except nx.NetworkXNoPath:
            return None
    
    def suggest_learning_focus(self):
        """Suggest what the system should focus on learning next"""
        suggestions = []
        
        # Find isolated concepts
        isolated = [concept for concept, degree in self.concept_graph.degree() if degree == 0]
        if isolated:
            suggestions.append(f"Connect isolated concepts: {isolated[:3]}")
        
        # Find important but shallow concepts
        shallow_important = []
        for concept, data in self.concept_graph.nodes(data=True):
            if data.get('importance', 0.5) > 0.7 and data.get('depth', 0.5) < 0.6:
                shallow_important.append(concept)
        
        if shallow_important:
            suggestions.append(f"Deepen important concepts: {shallow_important[:3]}")
        
        # Find underexplored categories
        categories = defaultdict(int)
        for concept, data in self.concept_graph.nodes(data=True):
            categories[data.get('category', 'unknown')] += 1
        
        underrepresented = [cat for cat, count in categories.items() if count < 3 and cat != 'unknown']
        if underrepresented:
            suggestions.append(f"Explore categories: {underrepresented}")
        
        # Find high mystery concepts
        mysterious = []
        for concept, data in self.concept_graph.nodes(data=True):
            if data.get('mystery', 0.5) > 0.8:
                mysterious.append(concept)
        
        if mysterious:
            suggestions.append(f"Investigate mysterious concepts: {mysterious[:3]}")
        
        return suggestions

# Enhanced Interactive Interface
class ConceptualMindInterface:
    def __init__(self, mind):
        self.mind = mind
        self.command_history = []
    
    def run_interactive_session(self):
        """Run interactive session with the conceptual mind"""
        print("""
🌟 Welcome to the Enhanced Conceptual Mind Interface!

Available commands:
• learn [concept] - Discover a new concept
• query [question] - Ask about existing knowledge  
• connect [concept1] [concept2] - Explore connection between concepts
• cluster [concept] - Explore concept cluster
• path [concept1] [concept2] - Find path between concepts
• add [concept] [definition] - Add your own concept
• status - Show current knowledge state
• suggest - Get learning suggestions
• start - Start continuous learning
• stop - Stop continuous learning  
• save - Save knowledge to disk
• help - Show this help
• exit - Exit interface

Type your command:
        """)
        
        while True:
            try:
                user_input = input("🧠> ").strip()
                if not user_input:
                    continue
                
                self.command_history.append(user_input)
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == "exit":
                    self.mind.stop_learning()
                    print("👋 Goodbye!")
                    break
                elif command == "learn" and len(parts) > 1:
                    concept = " ".join(parts[1:])
                    result = self.mind.discover_concept_with_llm(concept)
                    print(f"✨ Learned about '{concept}':")
                    print(result[:300] + "..." if len(str(result)) > 300 else result)
                
                elif command == "query" and len(parts) > 1:
                    query = " ".join(parts[1:])
                    result = self.mind.query_knowledge(query)
                    print(f"🔍 Query results for '{query}':")
                    for concept_info in result['relevant_concepts']:
                        print(f"• {concept_info['name']}: {concept_info['definition'][:100]}...")
                    if result['insights']:
                        print(f"💡 Insights: {result['insights'][0][:200]}...")
                
                elif command == "connect" and len(parts) > 2:
                    concept1, concept2 = parts[1], parts[2]
                    result = asyncio.run(self.mind.explore_concept_relationship_async(concept1, concept2))
                    print(f"🔗 Connection between '{concept1}' and '{concept2}':")
                    print(result[:300] + "..." if result and len(result) > 300 else result)
                
                elif command == "cluster" and len(parts) > 1:
                    concept = parts[1]
                    result = self.mind.explore_concept_cluster(concept)
                    if result:
                        print(f"🕸️ Cluster around '{concept}':")
                        print(f"Size: {result['cluster_size']} concepts")
                        print(f"Connections: {result['connections']}")
                        print(f"Categories: {dict(result['categories'])}")
                
                elif command == "path" and len(parts) > 2:
                    concept1, concept2 = parts[1], parts[2]
                    result = self.mind.get_concept_path(concept1, concept2)
                    if result:
                        print(f"🛤️ Path from '{concept1}' to '{concept2}':")
                        print(" → ".join(result['path']))
                    else:
                        print("❌ No path found between concepts")
                
                elif command == "add" and len(parts) > 1:
                    concept = parts[1]
                    definition = " ".join(parts[2:]) if len(parts) > 2 else None
                    result = self.mind.add_user_concept(concept, definition)
                    print(f"✅ Added concept '{concept}'")
                
                elif command == "status":
                    self.mind.visualize_knowledge(detailed=True)
                
                elif command == "suggest":
                    suggestions = self.mind.suggest_learning_focus()
                    print("💡 Learning suggestions:")
                    for suggestion in suggestions:
                        print(f"• {suggestion}")
                
                elif command == "start":
                    self.mind.start_continuous_learning()
                    print("🚀 Continuous learning started!")
                
                elif command == "stop":
                    self.mind.stop_learning()
                    print("⏹️ Learning stopped")
                
                elif command == "save":
                    self.mind._save_persistent_data()
                    print("💾 Knowledge saved")
                
                elif command == "help":
                    print("""
Available commands:
• learn [concept] - Discover a new concept
• query [question] - Ask about existing knowledge  
• connect [concept1] [concept2] - Explore connection
• cluster [concept] - Explore concept cluster
• path [concept1] [concept2] - Find path between concepts
• add [concept] [definition] - Add your own concept
• status - Show current knowledge state
• suggest - Get learning suggestions
• start - Start continuous learning
• stop - Stop continuous learning  
• save - Save knowledge to disk
• exit - Exit interface
                    """)
                
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                self.mind.stop_learning()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# Main execution
def main():
    """Main function to run the enhanced conceptual mind"""
    print("🚀 Starting Enhanced Conceptual Mind System")
    
    # Initialize the mind (you need to provide your Gemini API key)
    api_key = input("Enter your Gemini API key (or press Enter to use demo mode): ").strip()
    if not api_key:
        api_key = "YOUR_API_KEY_HERE"  # Demo mode
    
    mind = LLMEnhancedConceptualMind(
        brain_name="EnhancedGeminiMind",
        api_key=api_key,
        persist_data=True
    )
    
    # Show initial state
    mind.visualize_knowledge()
    
    # Start interactive interface
    interface = ConceptualMindInterface(mind)
    interface.run_interactive_session()

if __name__ == "__main__":
    main()
