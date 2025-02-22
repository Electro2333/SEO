from flask import Flask, request, jsonify, render_template
from seo_generator import Config, SEOParams, SEOContentGenerator, CONTENT_TYPES, ContentParameters
import os
from dotenv import load_dotenv
import asyncio
from functools import wraps
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please add it to your .env file.")

app = Flask(__name__)

# Initialize the SEO generator with default config
config = Config(
    model="mistralai/mistral-7b-instruct:free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    site_url=os.getenv("SITE_URL", "https://yoursiteurl.com"),
    site_name=os.getenv("SITE_NAME", "Your Site Name")
)

seo_params = SEOParams()
generator = SEOContentGenerator(config, seo_params)

# Helper function to run async functions
def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/categories')
def get_categories():
    """Return available content categories"""
    return jsonify({
        "categories": [
            {
                "name": cat.name,
                "icon": cat.icon,
                "key": cat.key,
                "structure": cat.structure
            }
            for cat in SEOContentGenerator.CATEGORIES.values()
        ]
    })

@app.route('/content-options')
def get_content_options():
    """Return available content options."""
    return jsonify({
        "content_types": {
            key: {
                "name": ct.name,
                "icon": ct.icon,
                "tones": ct.tones,
                "audiences": ct.audiences,
                "languages": ct.languages
            }
            for key, ct in CONTENT_TYPES.items()
        }
    })

@app.route('/generate', methods=['POST'])
@async_route
async def generate_content():
    try:
        data = request.get_json()
        
        # Update the model in config if specified
        if 'model' in data:
            config.model = data['model']
        
        # Create content parameters
        parameters = ContentParameters(
            content_type=data.get('content_type', 'article'),
            target_audience=data.get('target_audience', 'general'),
            tone=data.get('tone', 'professional'),
            language=data.get('language', 'en'),
            word_count=int(data.get('word_count', 2000))
        )
        
        # Generate content
        result = await generator.generate_seo_content(
            topic=data.get('topic'),
            keywords=data.get('keywords', []),
            parameters=parameters
        )
        
        return jsonify({"status": "success", "data": result})
        
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """Return list of available models"""
    return jsonify({
        "default": config.model,
        "available_models": [
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "deepseek/deepseek-r1:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen/qwen-vl-plus:free",
            "nvidia/llama-3.1-nemotron-70b-instruct:free",
            "sophosympatheia/rogue-rose-103b-v0.2:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "gryphe/mythomax-l2-13b:free"
        ]
    })

@app.route('/update-config', methods=['POST'])
def update_config():
    """Update generator configuration"""
    try:
        data = request.get_json()
        
        # Update config with new values
        if 'model' in data:
            config.model = data['model']
        if 'site_url' in data:
            config.site_url = data['site_url']
        if 'site_name' in data:
            config.site_name = data['site_name']
        
        # Reinitialize generator with updated config
        global generator
        generator = SEOContentGenerator(config, seo_params)
        
        return jsonify({
            "status": "success",
            "message": "Configuration updated successfully",
            "current_config": {
                "model": config.model,
                "site_url": config.site_url,
                "site_name": config.site_name
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api-status')
def api_status():
    return jsonify({
        "api_key_configured": bool(config.api_key),
        "base_url": config.base_url,
        "current_model": config.model
    })

if __name__ == '__main__':
    hyper_config = HyperConfig()
    hyper_config.bind = ["0.0.0.0:5000"]
    hyper_config.use_reloader = True
    asyncio.run(serve(app, hyper_config)) 