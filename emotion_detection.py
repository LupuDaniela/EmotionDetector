from transformers import pipeline
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import matplotlib.pyplot as plt
import sys
from googletrans import Translator

class EmotionDetector:
    EMOTION_RO = {
        'admiration': 'admirație', 'amusement': 'amuzament', 'anger': 'furie',
        'annoyance': 'enervare', 'approval': 'aprobare', 'caring': 'grijă',
        'confusion': 'confuzie', 'curiosity': 'curiozitate', 'desire': 'dorință',
        'disappointment': 'dezamăgire', 'disapproval': 'dezaprobare',
        'disgust': 'dezgust', 'embarrassment': 'jenă', 'excitement': 'entuziasm',
        'fear': 'frică', 'gratitude': 'recunoștință', 'grief': 'durere',
        'joy': 'bucurie', 'love': 'dragoste', 'nervousness': 'nervozitate',
        'optimism': 'optimism', 'pride': 'mândrie', 'realization': 'realizare',
        'relief': 'ușurare', 'remorse': 'remușcare', 'sadness': 'tristețe',
        'surprise': 'surpriză', 'neutral': 'neutru'
    }

    def __init__(self, db_config):
        print("Initializing Emotion Detector...")
        print("Loading RoBERTa GoEmotions...")
        try:
            self.classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
            self.translator = Translator()
            print("Model loaded! 28 emotions ready.")
            print("Translation enabled (RO → EN)\n")
        except Exception as e:
            print("Error loading model:", e)
            sys.exit(1)
        self.db_config = db_config
        self.init_database()

    def init_database(self):
        print("Connecting to PostgreSQL...")
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    message TEXT NOT NULL,
                    message_en TEXT,
                    primary_emotion VARCHAR(50) NOT NULL,
                    primary_emotion_ro VARCHAR(50),
                    primary_score FLOAT NOT NULL,
                    all_scores JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations (timestamp DESC)
            """)
            conn.commit()
            cursor.close()
            conn.close()
            print("Database connected!\n")
        except psycopg2.OperationalError as e:
            print("PostgreSQL connection error:", e)
            sys.exit(1)

    def detect_language(self, text):
        try:
            detection = self.translator.detect(text)
            return detection.lang
        except:
            return 'en'

    def translate_to_english(self, text):
        try:
            translation = self.translator.translate(text, src='auto', dest='en')
            return translation.text
        except Exception as e:
            return text

    def detect_emotions(self, message):
        original_message = message
        detected_lang = self.detect_language(message)
        
        if detected_lang != 'en':
            print(f" Language: {detected_lang.upper()}")
            message_en = self.translate_to_english(message)
            print(f"  🔄 Translation: {message_en}\n")
        else:
            message_en = message
        
        word_count = len(message_en.split())
        if word_count < 5:
            print("💡 Tip: Add more details for better accuracy!")
        
        emotion_keywords = {
            'anger': ['fight', 'angry', 'mad', 'furious', 'hate', 'pissed', 'annoyed', 'irritated', 'rage', 'argued', 'argument'],
            'sadness': ['sad', 'depressed', 'crying', 'broke up', 'lost', 'died', 'death', 'lonely', 'heartbroken', 'miserable', 'passed away'],
            'joy': ['happy', 'excited', 'great', 'amazing', 'wonderful', 'thrilled', 'delighted', 'pleased', 'fantastic'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'panic', 'frightened'],
            'love': ['love', 'adore', 'cherish', 'affection', 'romantic', 'crush', 'boyfriend', 'girlfriend'],
            'disgust': ['disgusting', 'gross', 'revolting', 'nasty', 'sick', 'vomit']
        }
        
        message_lower = message_en.lower()
        detected_keywords = []
        
        for emotion, keywords in emotion_keywords.items():
            if any(word in message_lower for word in keywords):
                detected_keywords.append(emotion)
        
        results = self.classifier(message_en)[0]
        sorted_result = sorted(results, key=lambda x: x['score'], reverse=True)
        primary = sorted_result[0]
        emotion_en_label = primary['label']
        score = primary['score']
        
        if detected_keywords and score < 0.50:
            print(f"  💡 Keywords: {', '.join(detected_keywords)}")
            
            for keyword_emotion in detected_keywords:
                matching = [r for r in sorted_result[:10] if keyword_emotion in r['label']]
                if matching and matching[0]['score'] > 0.05:
                    print(f"  → Adjusted: {matching[0]['label']}\n")
                    primary = matching[0]
                    emotion_en_label = primary['label']
                    score = primary['score']
                    break
        
        if score < 0.35:
            print(f"  ⚠️  Low confidence ({score:.1%})\n")
        
        emotion_ro = self.EMOTION_RO.get(emotion_en_label, emotion_en_label)
        
        top_5 = [
            {
                'emotion_en': r['label'],
                'emotion_ro': self.EMOTION_RO.get(r['label'], r['label']),
                'score': r['score']
            }
            for r in sorted_result[:5]
        ]
        
        all_scores = {r['label']: r['score'] for r in results}
        
        result = {
            'text': original_message,
            'text_en': message_en if detected_lang != 'en' else None,
            'primary_emotion': emotion_en_label,
            'primary_emotion_ro': emotion_ro,
            'primary_score': float(score),
            'top_5': top_5,
            'all_scores': all_scores,
            'timestamp': datetime.now()
        }
        
        self.save_to_database(result)
        return result

    def save_to_database(self, result):
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (message, message_en, primary_emotion, primary_emotion_ro, primary_score, all_scores, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                result['text'],
                result.get('text_en'),
                result['primary_emotion'],
                result['primary_emotion_ro'],
                result['primary_score'],
                Json(result['all_scores']),
                result['timestamp']
            ))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print("Save error:", e)

    def print_result(self, result):
        emotion = result['primary_emotion']
        emotion_ro = result['primary_emotion_ro']
        score = result['primary_score']
        
        print(f"\n{'='*70}")
        print(f"Emotion: {emotion} ({emotion_ro})")
        print(f"Confidence: {score:.1%}")
        
        bar_length = int(score * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{bar}")
        
        print(f"\nTop 5 Emotions:")
        for idx, item in enumerate(result['top_5'], start=1):
            emo_ro = item['emotion_ro']
            emo_score = item['score']
            print(f"  {idx}. {emo_ro:15} → {emo_score:.1%}")
        print('='*70 + '\n')

    def get_stats(self):
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]
        cursor.execute("""
            SELECT primary_emotion_ro, COUNT(*) as count
            FROM conversations
            GROUP BY primary_emotion_ro
            ORDER BY count DESC
        """)
        distribution = cursor.fetchall()
        cursor.execute("""
            SELECT message, primary_emotion_ro, primary_score, timestamp
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recent = cursor.fetchall()
        cursor.close()
        conn.close()
        return {'total': total, 'distribution': distribution, 'recent': recent}

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'='*70}")
        print(f"STATISTICS")
        print(f"{'='*70}")
        print(f"\nTotal messages: {stats['total']}")
        
        if stats['distribution']:
            print(f"\nEmotion Distribution:")
            max_count = max(d[1] for d in stats['distribution'])
            for emotion_ro, count in stats['distribution']:
                percentage = (count / stats['total']) * 100
                bar_length = int((count / max_count) * 40)
                bar = '█' * bar_length
                print(f"  {emotion_ro:15} [{count:3}] {bar} {percentage:5.1f}%")
        
        if stats['recent']:
            print(f"\nLast 10 messages:")
            for message, emotion_ro, score, timestamp in stats['recent']:
                time_str = timestamp.strftime("%H:%M:%S")
                msg_short = (message[:40] + '...') if len(message) > 40 else message
                print(f"  • {msg_short:43} → {emotion_ro:12} ({score:.0%}) [{time_str}]")
        print('='*70 + '\n')

    def plot_emotions(self):
        stats = self.get_stats()
        if not stats['distribution']:
            print("No data to plot yet.\n")
            return
        
        emotions = [d[0] for d in stats['distribution'][:10]]
        counts = [d[1] for d in stats['distribution'][:10]]
        colors_list = ['#667eea','#764ba2','#f093fb','#4facfe','#43e97b',
                      '#fa709a','#fee140','#30cfd0','#a8edea','#fed6e3']
        
        plt.figure(figsize=(14, 7))
        bars = plt.bar(emotions, counts, color=colors_list[:len(emotions)])
        plt.title('Emotion Distribution', fontsize=18, fontweight='bold')
        plt.xlabel('Emotions', fontsize=14)
        plt.ylabel('Number of Messages', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = 'emotion_chart.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as {filename}\n")
        try:
            plt.show()
        except:
            pass


def main():
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'emotion_db',
        'user': 'postgres',
        'password': '1q2w3e'
    }
    
    try:
        detector = EmotionDetector(db_config)
        print("="*70)
        print("EMOTION DETECTOR READY!")
        print("="*70)
        print("\nCommands:")
        print("  [write message] - Analyze emotion (RO/EN)")
        print("  stats           - Show statistics")
        print("  graph           - Plot chart")
        print("  quit            - Exit\n")
        
        message_count = 0
        
        while True:
            try:
                text = input("Your message: ").strip()
                
                if text.lower() in ['exit', 'quit', 'q']:
                    break
                elif text.lower() == 'stats':
                    detector.print_stats()
                    continue
                elif text.lower() == 'graph':
                    detector.plot_emotions()
                    continue
                elif not text:
                    print("Please enter a message.\n")
                    continue
                
                result = detector.detect_emotions(text)
                detector.print_result(result)
                message_count += 1
                
                if message_count == 5:
                    print("💡 Type 'stats' to see statistics!\n")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...\n")
                break
            except Exception as e:
                print("Error:", e, "\n")
        
        if message_count > 0:
            print("\nFinal Statistics:")
            detector.print_stats()
        
        print("Goodbye!\n")
        
    except Exception as e:
        print("Initialization error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()