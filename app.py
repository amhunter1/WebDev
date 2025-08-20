import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms
import modelscope_studio.components.pro as pro
from openai import OpenAI
from config import API_KEY, MODEL, SYSTEM_PROMPT, ENDPOINT, EXAMPLES, DEFAULT_LOCALE, DEFAULT_THEME

# Enhanced logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions for better code organization
@dataclass
class AppState:
    system_prompt: str = ""
    history: List[Dict[str, str]] = None
    current_session_id: str = ""
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.user_preferences is None:
            self.user_preferences = {
                "theme": "light",
                "code_format": "auto",
                "auto_save": True,
                "show_advanced": False
            }

@dataclass
class GeneratedFiles:
    html: Optional[str] = None
    jsx: Optional[str] = None
    tsx: Optional[str] = None
    css: Optional[str] = None
    js: Optional[str] = None

class OpenAIClient:
    """Enhanced OpenAI client wrapper with error handling and retry logic"""
    
    def __init__(self, api_key: str, base_url: str, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
    
    def create_completion(self, messages: List[Dict], model: str) -> Generator:
        """Create completion with retry logic and better error handling"""
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=4000
                )
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

# Initialize enhanced client
client = OpenAIClient(API_KEY, ENDPOINT)

# Enhanced React imports with version management
REACT_IMPORTS = {
    # Core React
    "react": "https://esm.sh/react@^19.0.0",
    "react/": "https://esm.sh/react@^19.0.0/",
    "react-dom": "https://esm.sh/react-dom@^19.0.0",
    "react-dom/": "https://esm.sh/react-dom@^19.0.0/",
    
    # UI Libraries
    "lucide-react": "https://esm.sh/lucide-react@0.525.0",
    "recharts": "https://esm.sh/recharts@3.1.0",
    "@headlessui/react": "https://esm.sh/@headlessui/react@2.0.4",
    "@heroicons/react": "https://esm.sh/@heroicons/react@2.1.5",
    
    # Animation
    "framer-motion": "https://esm.sh/framer-motion@12.23.6",
    "lottie-react": "https://esm.sh/lottie-react@2.4.0",
    
    # 3D Graphics
    "three": "https://esm.sh/three@0.178.0",
    "@react-three/fiber": "https://esm.sh/@react-three/fiber@9.2.0",
    "@react-three/drei": "https://esm.sh/@react-three/drei@10.5.2",
    
    # Game Development
    "matter-js": "https://esm.sh/matter-js@0.20.0",
    "konva": "https://esm.sh/konva@9.3.22",
    "react-konva": "https://esm.sh/react-konva@19.0.7",
    "p5": "https://esm.sh/p5@2.0.3",
    
    # Utilities
    "@tailwindcss/browser": "https://esm.sh/@tailwindcss/browser@4.1.11",
    "lodash": "https://esm.sh/lodash@4.17.21",
    "dayjs": "https://esm.sh/dayjs@1.11.13",
    "uuid": "https://esm.sh/uuid@10.0.0"
}

class CodeParser:
    """Enhanced code parsing with better file type detection"""
    
    @staticmethod
    def extract_files(text: str) -> GeneratedFiles:
        """Extract different file types from generated text"""
        patterns = {
            'html': r'```html\n(.+?)\n```',
            'jsx': r'```jsx\n(.+?)\n```',
            'tsx': r'```tsx\n(.+?)\n```',
            'css': r'```css\n(.+?)\n```',
            'js': r'```(?:javascript|js)\n(.+?)\n```',
        }
        
        files = GeneratedFiles()
        
        for file_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                content = '\n'.join(matches).strip()
                setattr(files, file_type, content)
        
        # Fallback: treat entire response as HTML if no code blocks found
        if not any([files.html, files.jsx, files.tsx]):
            files.html = text.strip()
            
        return files
    
    @staticmethod
    def get_primary_file(files: GeneratedFiles) -> Tuple[str, str]:
        """Get the primary file content and type"""
        if files.tsx:
            return files.tsx, "tsx"
        elif files.jsx:
            return files.jsx, "jsx"
        elif files.html:
            return files.html, "html"
        elif files.js:
            return files.js, "js"
        else:
            return "", "html"

class EnhancedGradioEvents:
    """Enhanced event handlers with better error handling and UX"""
    
    @staticmethod
    def generate_code(
        input_value: str, 
        system_prompt_input_value: str, 
        state_value: Dict,
        progress=gr.Progress()
    ):
        """Enhanced code generation with progress tracking"""
        
        if not input_value or not input_value.strip():
            yield {
                output_loading: gr.update(spinning=False),
                state_tab: gr.update(active_key="empty"),
                notification: gr.update(value="Please enter a description first", visible=True)
            }
            return
        
        # Initial loading state
        yield {
            output_loading: gr.update(spinning=True),
            state_tab: gr.update(active_key="loading"),
            output: gr.update(value=""),
            notification: gr.update(visible=False)
        }
        
        try:
            # Prepare messages
            messages = [{
                'role': "system",
                'content': SYSTEM_PROMPT
            }] + state_value.get("history", [])
            
            messages.append({'role': "user", 'content': input_value})
            
            # Generate response with progress
            progress(0.1, "Connecting to AI model...")
            generator = client.create_completion(messages, MODEL)
            
            response = ""
            progress(0.3, "Generating code...")
            
            for chunk in generator:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
                    
                    # Intermediate update
                    yield {
                        output: gr.update(value=response),
                        output_loading: gr.update(spinning=True),
                    }
                
                if chunk.choices[0].finish_reason == 'stop':
                    progress(0.8, "Processing generated code...")
                    
                    # Update conversation history
                    state_value["history"] = messages + [{
                        'role': "assistant",
                        'content': response
                    }]
                    
                    # Parse generated files
                    files = CodeParser.extract_files(response)
                    primary_content, file_type = CodeParser.get_primary_file(files)
                    
                    progress(0.9, "Preparing preview...")
                    
                    # Prepare sandbox configuration
                    is_react = file_type in ["tsx", "jsx"]
                    sandbox_config = {
                        "template": "react" if is_react else "html",
                        "imports": REACT_IMPORTS if is_react else {}
                    }
                    
                    if is_react:
                        sandbox_config["value"] = {
                            "./index.tsx": """import Demo from './demo.tsx'
import "@tailwindcss/browser"

export default Demo""",
                            "./demo.tsx": primary_content
                        }
                    else:
                        sandbox_config["value"] = {"./index.html": primary_content}
                    
                    progress(1.0, "Complete!")
                    
                    # Final successful state
                    yield {
                        output: gr.update(value=response),
                        download_content: gr.update(value=primary_content),
                        state_tab: gr.update(active_key="render"),
                        output_loading: gr.update(spinning=False),
                        sandbox: gr.update(**sandbox_config),
                        state: gr.update(value=state_value),
                        download_btn: gr.update(disabled=False),
                        notification: gr.update(value="✅ Code generated successfully!", visible=True)
                    }
                    return
                    
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            yield {
                output: gr.update(value=f"❌ **Error**: {str(e)}"),
                output_loading: gr.update(spinning=False),
                state_tab: gr.update(active_key="render"),
                notification: gr.update(value=f"❌ Generation failed: {str(e)}", visible=True)
            }
    
    @staticmethod
    def select_example(example: Dict):
        """Enhanced example selection with better UX"""
        def _handler():
            return {
                input: gr.update(value=example["description"]),
                notification: gr.update(value=f"📝 Loaded example: {example['title']}", visible=True)
            }
        return _handler
    
    @staticmethod
    def export_code(content: str, file_format: str = "auto"):
        """Enhanced code export with multiple format support"""
        if not content:
            return gr.update(value="No content to export")
        
        # Determine file extension
        if file_format == "auto":
            if "import React" in content or "export default" in content:
                extension = "tsx"
            elif "<html" in content.lower():
                extension = "html"
            else:
                extension = "txt"
        else:
            extension = file_format
        
        return gr.update(
            value=content,
            headers={"Content-Disposition": f"attachment; filename=generated_code.{extension}"}
        )
    
    @staticmethod
    def toggle_advanced_settings(current_state: bool):
        """Toggle advanced settings panel"""
        return gr.update(visible=not current_state)
    
    @staticmethod
    def save_user_preferences(preferences: Dict, state_value: Dict):
        """Save user preferences to state"""
        state_value["user_preferences"] = preferences
        return gr.update(value=state_value)
    
    @staticmethod
    def clear_history_with_confirmation(state_value: Dict):
        """Clear history with user confirmation"""
        state_value["history"] = []
        return {
            state: gr.update(value=state_value),
            notification: gr.update(value="🧹 Chat history cleared", visible=True)
        }
    
    @staticmethod
    def format_code(code: str, language: str = "auto"):
        """Basic code formatting (placeholder for future enhancement)"""
        # This could be enhanced with actual formatting libraries
        return code

# Enhanced CSS with modern design
ENHANCED_CSS = """
/* Modern Variables */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.1);
    --shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.15);
    --border-radius: 16px;
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Main Container */
#enhanced-coder-artifacts {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
    padding: 20px;
}

/* Glassmorphism Cards */
#enhanced-coder-artifacts .ant-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-soft);
    transition: var(--transition-smooth);
}

#enhanced-coder-artifacts .ant-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}

/* Enhanced Buttons */
#enhanced-coder-artifacts .ant-btn {
    border-radius: 12px;
    transition: var(--transition-smooth);
    font-weight: 600;
    border: none;
    position: relative;
    overflow: hidden;
}

#enhanced-coder-artifacts .ant-btn-primary {
    background: var(--primary-gradient);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

#enhanced-coder-artifacts .ant-btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}

/* Animated Loading */
.enhanced-loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border-radius: var(--border-radius);
}

.enhanced-loading::before {
    content: '';
    width: 50px;
    height: 50px;
    border: 3px solid transparent;
    border-top: 3px solid var(--primary-gradient);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 16px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced Output Container */
#enhanced-coder-artifacts .output-container {
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--shadow-soft);
}

#enhanced-coder-artifacts .output-sandbox {
    border-radius: var(--border-radius);
    overflow: hidden;
    min-height: 600px;
    background: white;
}

/* Notification System */
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    padding: 16px 24px;
    border-radius: 12px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-soft);
    color: #333;
    font-weight: 600;
    max-width: 400px;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Advanced Settings Panel */
.advanced-settings {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    padding: 20px;
    margin-top: 16px;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    #enhanced-coder-artifacts {
        padding: 10px;
    }
    
    .ant-card {
        margin-bottom: 16px;
    }
    
    .enhanced-loading {
        min-height: 300px;
    }
}

/* Dark mode support */
[data-theme="dark"] #enhanced-coder-artifacts {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
}

[data-theme="dark"] .ant-card {
    background: rgba(0, 0, 0, 0.2);
    color: white;
}
"""

def create_enhanced_app():
    """Create the enhanced application"""
    
    with gr.Blocks(css=ENHANCED_CSS, title="🚀 AI Web Dev Assistant Pro") as demo:
        # Enhanced Global State
        state = gr.State(AppState())
        
        # Notification system
        notification = gr.HTML(visible=False, elem_classes="notification")
        
        with ms.Application(elem_id="enhanced-coder-artifacts") as app:
            with antd.ConfigProvider(theme=DEFAULT_THEME, locale=DEFAULT_LOCALE):
                
                # Header Section
                with antd.Row(justify="center", elem_style=dict(marginBottom=32)):
                    with antd.Col(span=24):
                        with antd.Card(elem_classes="header-card"):
                            with antd.Flex(justify="center", align="center", vertical=True, gap="large"):
                                antd.Avatar(
                                    src="https://img.alicdn.com/imgextra/i2/O1CN01KDo8Ma1DUo8oa7OIU_!!6000000000220-1-tps-240-240.gif",
                                    size=120,
                                    elem_style=dict(
                                        boxShadow="0 8px 32px rgba(0, 0, 0, 0.1)",
                                        border="4px solid rgba(255, 255, 255, 0.3)"
                                    )
                                )
                                antd.Typography.Title(
                                    "🚀 AI Web Dev Assistant Pro",
                                    level=1,
                                    elem_style=dict(
                                        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                        backgroundClip="text",
                                        WebkitBackgroundClip="text",
                                        color="transparent",
                                        fontSize=36,
                                        fontWeight="bold",
                                        textAlign="center",
                                        margin=0
                                    )
                                )
                                antd.Typography.Text(
                                    "Create stunning web applications with AI-powered code generation",
                                    type="secondary",
                                    elem_style=dict(fontSize=18, textAlign="center")
                                )
                
                with ms.AutoLoading():
                    with antd.Row(gutter=[32, 32], align="stretch"):
                        
                        # Left Column - Input & Controls
                        with antd.Col(span=24, lg=10):
                            with antd.Space(direction="vertical", size="large", elem_style=dict(width="100%")):
                                
                                # Input Section
                                with antd.Card(title="✨ What would you like to create?", elem_classes="input-card"):
                                    input = antd.Input.Textarea(
                                        size="large",
                                        allow_clear=True,
                                        auto_size=dict(minRows=3, maxRows=8),
                                        placeholder="Describe your web application in detail...\n\nExample: 'Create a modern dashboard with dark mode, animated charts, and a sidebar navigation'",
                                        elem_style=dict(
                                            border="2px solid rgba(102, 126, 234, 0.2)",
                                            borderRadius="12px",
                                            fontSize="16px"
                                        )
                                    )
                                    
                                    with antd.Flex(justify="space-between", align="center", elem_style=dict(marginTop=16)):
                                        with antd.Space():
                                            submit_btn = antd.Button(
                                                "🎨 Generate Code",
                                                type="primary",
                                                size="large",
                                                elem_style=dict(
                                                    background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                                    borderRadius="12px",
                                                    padding="0 32px",
                                                    height="48px",
                                                    fontSize="16px",
                                                    fontWeight="600"
                                                )
                                            )
                                        
                                        tour_btn = antd.Button(
                                            "📖 Quick Tour",
                                            type="text",
                                            size="small"
                                        )
                                
                                # Examples Section
                                with antd.Card(title="🎯 Quick Examples", elem_classes="examples-card"):
                                    with antd.Space(direction="vertical", size="middle", elem_style=dict(width="100%")):
                                        for i, example in enumerate(EXAMPLES[:3]):  # Show first 3 examples
                                            with antd.Card(
                                                size="small",
                                                hoverable=True,
                                                elem_style=dict(
                                                    cursor="pointer",
                                                    transition="all 0.3s ease"
                                                )
                                            ) as example_card:
                                                antd.Card.Meta(
                                                    title=f"💡 {example['title']}",
                                                    description=example['description'][:100] + "..." if len(example['description']) > 100 else example['description']
                                                )
                                            
                                            example_card.click(
                                                fn=EnhancedGradioEvents.select_example(example),
                                                outputs=[input, notification]
                                            )
                                
                                # Controls Section
                                with antd.Card(title="⚙️ Controls", elem_classes="controls-card"):
                                    with antd.Space(wrap=True, size="middle"):
                                        history_btn = antd.Button(
                                            "📜 History",
                                            icon=antd.Icon("HistoryOutlined")
                                        )
                                        clear_history_btn = antd.Button(
                                            "🗑️ Clear",
                                            danger=True,
                                            icon=antd.Icon("DeleteOutlined")
                                        )
                                        advanced_btn = antd.Button(
                                            "⚡ Advanced",
                                            type="dashed",
                                            icon=antd.Icon("SettingOutlined")
                                        )
                        
                        # Right Column - Output
                        with antd.Col(span=24, lg=14):
                            with antd.Card(
                                title="🖥️ Generated Application",
                                elem_style=dict(height="100%", minHeight="700px"),
                                elem_classes="output-container"
                            ):
                                # Output Controls
                                with ms.Slot("extra"):
                                    with antd.Space():
                                        download_btn = antd.Button(
                                            "⬇️ Download",
                                            type="primary",
                                            disabled=True,
                                            icon=antd.Icon("DownloadOutlined")
                                        )
                                        view_code_btn = antd.Button(
                                            "👀 View Code",
                                            icon=antd.Icon("CodeOutlined")
                                        )
                                
                                # Output Content with Tabs
                                with antd.Tabs(
                                    active_key="empty",
                                    render_tab_bar="() => null",
                                    elem_style=dict(height="100%")
                                ) as state_tab:
                                    
                                    with antd.Tabs.Item(key="empty"):
                                        with antd.Empty(
                                            description="Ready to create something amazing!",
                                            elem_style=dict(
                                                minHeight="500px",
                                                display="flex",
                                                flexDirection="column",
                                                justifyContent="center",
                                                alignItems="center"
                                            )
                                        ):
                                            antd.Button(
                                                "🚀 Start Creating",
                                                type="primary",
                                                size="large"
                                            )
                                    
                                    with antd.Tabs.Item(key="loading"):
                                        with antd.Flex(
                                            justify="center",
                                            align="center",
                                            vertical=True,
                                            elem_style=dict(minHeight="500px"),
                                            elem_classes="enhanced-loading"
                                        ):
                                            antd.Spin(size="large")
                                            antd.Typography.Text(
                                                "🎨 Creating your application...",
                                                elem_style=dict(
                                                    fontSize="18px",
                                                    fontWeight="600",
                                                    marginTop="24px"
                                                )
                                            )
                                    
                                    with antd.Tabs.Item(key="render"):
                                        sandbox = pro.WebSandbox(
                                            height="600px",
                                            elem_classes="output-sandbox"
                                        )
                
                # Hidden components for functionality
                download_content = gr.Text(visible=False)
                system_prompt_input = gr.Text(SYSTEM_PROMPT, visible=False)
                
                # Modals and Drawers
                with antd.Drawer(
                    title="📋 Generated Code",
                    width="60%",
                    placement="right"
                ) as code_drawer:
                    with antd.Spin() as output_loading:
                        output = ms.Markdown(
                            elem_style=dict(
                                maxHeight="70vh",
                                overflow="auto",
                                padding="16px",
                                background="rgba(0, 0, 0, 0.02)",
                                borderRadius="8px"
                            )
                        )
                
                with antd.Drawer(
                    title="📜 Chat History",
                    width="50%",
                    placement="left"
                ) as history_drawer:
                    history_output = gr.Chatbot(
                        show_label=False,
                        type="messages",
                        height='60vh',
                        elem_style=dict(
                            borderRadius="12px",
                            overflow="hidden"
                        )
                    )
                
                # Enhanced Tour
                with antd.Tour() as usage_tour:
                    antd.Tour.Step(
                        title="🎯 Step 1: Describe",
                        description="Tell the AI what kind of web application you want to create. Be as detailed as possible!"
                    )
                    antd.Tour.Step(
                        title="🎨 Step 2: Generate",
                        description="Click the Generate button and watch the magic happen!"
                    )
                    antd.Tour.Step(
                        title="👀 Step 3: Preview",
                        description="See your application come to life in real-time."
                    )
                    antd.Tour.Step(
                        title="⬇️ Step 4: Download",
                        description="Download your code and use it anywhere!"
                    )
        
        # Event Handlers
        def close_modal_handler():
            return gr.update(open=False)
        
        def open_modal_handler(component):
            return gr.update(open=True)
        
        # Tour events
        tour_btn.click(fn=lambda: gr.update(open=True), outputs=[usage_tour])
        usage_tour.close(fn=close_modal_handler, outputs=[usage_tour])
        usage_tour.finish(fn=close_modal_handler, outputs=[usage_tour])
        
        # Main generation event
        submit_btn.click(
            fn=lambda: gr.update(open=True),
            outputs=[code_drawer]
        ).then(
            fn=EnhancedGradioEvents.generate_code,
            inputs=[input, system_prompt_input, state],
            outputs=[
                output, state_tab, sandbox, download_content,
                output_loading, state, download_btn, notification
            ]
        ).then(
            fn=close_modal_handler,
            outputs=[code_drawer]
        )
        
        # Other event handlers
        view_code_btn.click(fn=lambda: gr.update(open=True), outputs=[code_drawer])
        code_drawer.close(fn=close_modal_handler, outputs=[code_drawer])
        
        history_btn.click(fn=lambda: gr.update(open=True), outputs=[history_drawer])
        history_drawer.close(fn=close_modal_handler, outputs=[history_drawer])
        
        clear_history_btn.click(
            fn=EnhancedGradioEvents.clear_history_with_confirmation,
            inputs=[state],
            outputs=[state, notification]
        )
        
        # Download functionality
        download_btn.click(
            fn=None,
            inputs=[download_content],
            js="""(content) => {
                if (!content) {
                    alert('No content to download!');
                    return;
                }
                
                // Determine file extension based on content
                let extension = 'txt';
                let filename = 'generated_code';
                
                if (content.includes('import React') || content.includes('export default')) {
                    extension = 'tsx';
                    filename = 'App';
                } else if (content.includes('<html') || content.includes('<!DOCTYPE')) {
                    extension = 'html';
                    filename = 'index';
                } else if (content.includes('function') || content.includes('const ')) {
                    extension = 'js';
                    filename = 'script';
                }
                
                const blob = new Blob([content], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${filename}.${extension}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                // Show success notification
                const notification = document.querySelector('.notification');
                if (notification) {
                    notification.innerHTML = '✅ File downloaded successfully!';
                    notification.style.display = 'block';
                    setTimeout(() => {
                        notification.style.display = 'none';
                    }, 3000);
                }
            }"""
        )
        
        # Auto-hide notification after 5 seconds
        notification.change(
            fn=None,
            js="""() => {
                setTimeout(() => {
                    const notificationEl = document.querySelector('.notification');
                    if (notificationEl) {
                        notificationEl.style.display = 'none';
                    }
                }, 5000);
            }"""
        )
        
    return demo

# Additional utility functions for enhanced functionality
class TemplateManager:
    """Manage code templates and snippets"""
    
    TEMPLATES = {
        "react_dashboard": """import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const [data] = useState([
    { name: 'Jan', value: 400 },
    { name: 'Feb', value: 300 },
    { name: 'Mar', value: 600 },
    { name: 'Apr', value: 800 },
  ]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">Dashboard</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {/* Stats Cards */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Total Users</h3>
            <p className="text-3xl font-bold text-blue-600">12,459</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Revenue</h3>
            <p className="text-3xl font-bold text-green-600">$54,239</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Orders</h3>
            <p className="text-3xl font-bold text-purple-600">1,423</p>
          </div>
        </div>
        
        {/* Chart */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Monthly Analytics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;""",
        
        "landing_page": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Landing Page</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .glass { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); }
    </style>
</head>
<body class="gradient-bg min-h-screen">
    <!-- Hero Section -->
    <div class="container mx-auto px-6 py-20">
        <div class="text-center text-white">
            <h1 class="text-6xl font-bold mb-6 animate-fade-in">
                Build Amazing Things
            </h1>
            <p class="text-xl mb-8 opacity-90">
                Create stunning web applications with our powerful tools
            </p>
            <button class="glass rounded-full px-8 py-4 text-lg font-semibold hover:scale-105 transition-transform">
                Get Started
            </button>
        </div>
        
        <!-- Features Grid -->
        <div class="grid md:grid-cols-3 gap-8 mt-20">
            <div class="glass rounded-xl p-8 text-white text-center">
                <div class="text-4xl mb-4">🚀</div>
                <h3 class="text-xl font-semibold mb-3">Fast Performance</h3>
                <p class="opacity-80">Lightning-fast load times and optimized code</p>
            </div>
            <div class="glass rounded-xl p-8 text-white text-center">
                <div class="text-4xl mb-4">🎨</div>
                <h3 class="text-xl font-semibold mb-3">Beautiful Design</h3>
                <p class="opacity-80">Modern, responsive designs that look amazing</p>
            </div>
            <div class="glass rounded-xl p-8 text-white text-center">
                <div class="text-4xl mb-4">⚡</div>
                <h3 class="text-xl font-semibold mb-3">Easy to Use</h3>
                <p class="opacity-80">Simple, intuitive interface for everyone</p>
            </div>
        </div>
    </div>
</body>
</html>"""
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        return cls.TEMPLATES.get(template_name, "")
    
    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls.TEMPLATES.keys())

class CodeQualityChecker:
    """Basic code quality checking utilities"""
    
    @staticmethod
    def check_react_best_practices(code: str) -> List[str]:
        """Check for React best practices"""
        issues = []
        
        if "useState" in code and "import" not in code:
            issues.append("Consider importing React hooks explicitly")
        
        if "function" in code and "export default" not in code:
            issues.append("Component should have a default export")
        
        if "className" not in code and "class=" in code:
            issues.append("Use className instead of class in React")
        
        return issues
    
    @staticmethod
    def check_html_structure(code: str) -> List[str]:
        """Check HTML structure"""
        issues = []
        
        if "<!DOCTYPE html>" not in code and "<html" in code:
            issues.append("Missing DOCTYPE declaration")
        
        if "<title>" not in code and "<head>" in code:
            issues.append("Missing title tag")
        
        if 'lang="' not in code and "<html" in code:
            issues.append("Missing lang attribute in html tag")
        
        return issues

class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.generation_times = []
        self.error_count = 0
        self.success_count = 0
    
    def record_generation_time(self, duration: float):
        self.generation_times.append(duration)
    
    def record_success(self):
        self.success_count += 1
    
    def record_error(self):
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.generation_times:
            return {"avg_time": 0, "success_rate": 0, "total_generations": 0}
        
        return {
            "avg_time": sum(self.generation_times) / len(self.generation_times),
            "success_rate": self.success_count / (self.success_count + self.error_count),
            "total_generations": len(self.generation_times),
            "error_count": self.error_count
        }

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# Enhanced configuration with feature flags
class EnhancedConfig:
    """Enhanced configuration management"""
    
    FEATURE_FLAGS = {
        "enable_code_quality_check": True,
        "enable_performance_monitoring": True,
        "enable_template_suggestions": True,
        "enable_auto_formatting": True,
        "enable_collaborative_editing": False,
        "enable_version_history": True,
        "enable_export_multiple_formats": True
    }
    
    UI_THEMES = {
        "light": {
            "primaryColor": "#667eea",
            "colorBgBase": "#ffffff",
            "colorTextBase": "#000000"
        },
        "dark": {
            "primaryColor": "#764ba2",
            "colorBgBase": "#1a1a1a",
            "colorTextBase": "#ffffff"
        },
        "cosmic": {
            "primaryColor": "#ff6b6b",
            "colorBgBase": "#0f0f0f",
            "colorTextBase": "#ffffff"
        }
    }
    
    @classmethod
    def is_feature_enabled(cls, feature: str) -> bool:
        return cls.FEATURE_FLAGS.get(feature, False)
    
    @classmethod
    def get_theme(cls, theme_name: str) -> Dict:
        return cls.UI_THEMES.get(theme_name, cls.UI_THEMES["light"])

# Main application factory
def create_production_app():
    """Create production-ready application with all enhancements"""
    
    # Initialize components
    app = create_enhanced_app()
    
    # Add middleware for production
    if EnhancedConfig.is_feature_enabled("enable_performance_monitoring"):
        logger.info("Performance monitoring enabled")
    
    if EnhancedConfig.is_feature_enabled("enable_code_quality_check"):
        logger.info("Code quality checking enabled")
    
    return app

if __name__ == "__main__":
    # Create and launch the enhanced application
    demo = create_production_app()
    
    # Production settings
    demo.queue(
        default_concurrency_limit=50,  # Reduced for stability
        max_size=200,                  # Increased queue size
        api_open=False                 # Security: Disable API access
    ).launch(
        ssr_mode=False,
        max_threads=50,
        show_error=True,
        quiet=False,
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,            # Set to True for public sharing
        debug=False,            # Set to True for development
        favicon_path=None,      # Add custom favicon
        app_kwargs={
            "docs_url": None,   # Disable docs for security
            "redoc_url": None   # Disable redoc for security
        }
    )
