
import gradio as gr
import os
import socket
import time
import numpy as np
from contextlib import closing

from utils.yolo_detection import YOLOv12Detector
from utils.gemini_captioning import GeminiCaptioner
from utils.gemini_vqa import GeminiVQA


# ==================================================
# 1. LOAD GEMINI API KEY
# ==================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    key_file = "utils/gemini_api_key.txt"

    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            GEMINI_API_KEY = f.read().strip()

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Gemini API key not found. Configure GEMINI_API_KEY "
        "or provide utils/gemini_api_key.txt."
    )


# ==================================================
# 2. TAERO AI - CINEMATIC MULTICOLOUR THEME
# ==================================================

contrast_css = """

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #090A10;
    --surface: #12141F;
    --surface2: #191B29;
    --white: #FFFFFF;
    --muted: #A5A8BB;

    --purple: #A855F7;
    --blue: #387BFF;
    --cyan: #00D9FF;
    --pink: #FF3D9A;
    --red: #F43F5E;
    --orange: #FF8738;
    --gold: #FFC247;
    --green: #00D6A3;

    --border: rgba(255,255,255,0.09);
}


/* ================================================
   GLOBAL BACKGROUND
================================================ */

body {
    background: var(--bg) !important;
}

.gradio-container {
    font-family: 'Inter', sans-serif !important;

    background:
        radial-gradient(
            ellipse at 5% 5%,
            rgba(120,50,255,0.16),
            transparent 32%
        ),
        radial-gradient(
            ellipse at 95% 10%,
            rgba(0,160,255,0.13),
            transparent 30%
        ),
        radial-gradient(
            ellipse at 50% 65%,
            rgba(255,40,130,0.07),
            transparent 38%
        ),
        linear-gradient(135deg, #090A10, #10111B 55%, #090A10) !important;

    color: var(--white) !important;

    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;

    margin: 0 !important;
    padding: 24px 40px !important;

    box-sizing: border-box !important;
}

.gradio-container .main,
.gradio-container .wrap {
    max-width: 100% !important;
}


/* ================================================
   HEADER - CINEMATIC HERO SECTION
================================================ */

#app-header {
    position: relative;
    overflow: hidden;
    isolation: isolate;

    text-align: center;

    padding: 52px 25px;
    margin: 10px 0 30px 0;

    border-radius: 23px;
    border: 1px solid rgba(255,255,255,0.13);

    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(168,85,247,0.42),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(0,190,255,0.32),
            transparent 35%
        ),
        radial-gradient(
            circle at 70% 100%,
            rgba(255,40,140,0.28),
            transparent 40%
        ),
        linear-gradient(
            115deg,
            #17102D,
            #17152F,
            #101C35,
            #25112B
        );

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        0 0 35px rgba(120,70,255,0.12);
}

#app-header::before {
    content: "";
    position: absolute;
    z-index: -1;

    width: 280px;
    height: 280px;

    top: -170px;
    right: 8%;

    border-radius: 50%;

    background: rgba(0,217,255,0.18);
    filter: blur(45px);
}

#app-header::after {
    content: "";
    position: absolute;
    z-index: -1;

    width: 250px;
    height: 250px;

    bottom: -180px;
    left: 8%;

    border-radius: 50%;

    background: rgba(255,61,154,0.18);
    filter: blur(45px);
}

#app-header .brand-mark {
    width: 65px;
    height: 65px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin: 0 auto 16px auto;

    border-radius: 19px;

    background: linear-gradient(
        135deg,
        #A855F7,
        #387BFF,
        #00D9FF,
        #FF3D9A
    );

    color: white;
    font-size: 33px;

    box-shadow:
        0 8px 30px rgba(120,80,255,0.40),
        0 0 35px rgba(0,217,255,0.20);
}

#app-header .tagline {
    color: #64E6FF;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3.5px;
    margin-bottom: 17px;
}

#app-header h1 {
    color: white;

    font-family: 'Inter', sans-serif !important;
    font-size: clamp(48px, 5vw, 62px);
    font-weight: 800;

    letter-spacing: -2.5px;
    line-height: 1.12;
    margin: 0 0 12px 0;

    background: linear-gradient(
        90deg,
        #FFFFFF,
        #D9B5FF,
        #76E8FF,
        #FF9ED1,
        #FFFFFF
    );

    background-size: 200% auto;

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

#app-header .subtitle {
    color: #FF91C9;

    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2.5px;

    margin-bottom: 16px;
}

#app-header p {
    color: #D5D7E7;

    font-size: 16px;
    font-weight: 400;
    line-height: 1.85;
    letter-spacing: 0.1px;

    max-width: 720px;
    margin: auto;
}


/* ================================================
   SECTION HEADINGS - IMPROVED READABILITY
================================================ */

.section-title {
    margin: 0 0 3px 0 !important;
    padding: 0 !important;
}

.section-title h3 {
    color: #FFFFFF !important;

    font-family: 'Inter', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;

    letter-spacing: -0.7px !important;
    line-height: 1.35 !important;

    margin: 0 0 3px 0 !important;
    padding: 0 !important;
}

.section-note {
    color: #B8BBD0;

    font-size: 13px;
    font-weight: 400;
    line-height: 1.6;
    letter-spacing: 0.1px;

    margin-top: 0;
    margin-bottom: 12px;
}


/* ================================================
   PREMIUM GLASS CARDS
================================================ */

.panel {
    position: relative;
    overflow: hidden;

    background: linear-gradient(
        145deg,
        rgba(25,27,43,0.98),
        rgba(16,18,30,0.98)
    ) !important;

    border: 1px solid var(--border) !important;
    border-radius: 19px !important;

    padding: 21px !important;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.24),
        inset 0 1px 0 rgba(255,255,255,0.025);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

.panel:hover {
    transform: translateY(-3px);

    border-color: rgba(255,255,255,0.20) !important;

    box-shadow:
        0 18px 45px rgba(0,0,0,0.35);
}

.panel h3 {
    color: #FFFFFF !important;
    font-size: 26px !important;
    font-weight: 800 !important;
}


/* Individual card accent colours */

.upload-panel {
    border-top: 3px solid var(--purple) !important;
    box-shadow:
        0 12px 35px rgba(0,0,0,0.24),
        inset 0 3px 20px rgba(168,85,247,0.045);
}

.detection-panel {
    border-top: 3px solid var(--cyan) !important;
    box-shadow:
        0 12px 35px rgba(0,0,0,0.24),
        inset 0 3px 20px rgba(0,217,255,0.045);
}

.question-panel {
    border-top: 3px solid var(--pink) !important;
    box-shadow:
        0 12px 35px rgba(0,0,0,0.24),
        inset 0 3px 20px rgba(255,61,154,0.045);

    margin-top: 24px !important;
}


/* ================================================
   IMAGE UPLOAD AND DETECTION
================================================ */

.image-box {
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;

    overflow: hidden !important;

    background: #10121D !important;
}

.upload-panel .image-box {
    border-color: rgba(168,85,247,0.45) !important;
}

.detection-panel .image-box {
    border-color: rgba(0,217,255,0.40) !important;
}

.upload-panel .image-box .icon {
    color: var(--purple) !important;
}

.detection-panel .image-box .icon {
    color: var(--cyan) !important;
}


/* ================================================
   CAPTION / DESCRIPTION
================================================ */

#caption-panel {
    max-height: 260px !important;
    min-height: 110px !important;

    overflow-y: auto !important;
    overflow-x: hidden !important;

    background: linear-gradient(
        135deg,
        rgba(255,135,56,0.09),
        rgba(25,27,43,0.95)
    ) !important;

    border: 1px solid rgba(255,194,71,0.32) !important;
    border-left: 4px solid var(--gold) !important;

    border-radius: 13px !important;
    padding: 15px !important;

    color: #F7F2E9 !important;
    line-height: 1.8 !important;

    scrollbar-width: thin;
    scrollbar-color: var(--gold) #171927;
}

#caption-panel h3,
#caption-panel h4 {
    color: #FFD477 !important;
}

#caption-panel p,
#caption-panel li {
    color: #E8E7EF !important;
    font-size: 15px !important;
}


/* ================================================
   QUESTION AND ANSWER
================================================ */

#answer-panel {
    max-height: 220px !important;
    min-height: 75px !important;

    overflow-y: auto !important;
    overflow-x: hidden !important;

    background: linear-gradient(
        135deg,
        rgba(255,61,154,0.09),
        rgba(25,27,43,0.95)
    ) !important;

    border: 1px solid rgba(255,61,154,0.34) !important;
    border-left: 4px solid var(--pink) !important;

    border-radius: 13px !important;
    padding: 15px !important;

    color: #F7ECF4 !important;
    line-height: 1.8 !important;

    scrollbar-width: thin;
    scrollbar-color: var(--pink) #171927;
}

#answer-panel h3,
#answer-panel h4 {
    color: #FF8BC5 !important;
}

#answer-panel p,
#answer-panel li {
    color: #EDE4F0 !important;
    font-size: 15px !important;
}


/* ================================================
   TEXTBOX
================================================ */

.textbox textarea,
.textbox input {
    background: #10121D !important;
    color: white !important;

    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;

    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;

    transition: 0.2s ease;
}

.textbox textarea:focus,
.textbox input:focus {
    border-color: var(--pink) !important;

    box-shadow:
        0 0 0 2px rgba(255,61,154,0.15),
        0 0 15px rgba(255,61,154,0.10) !important;
}

.textbox textarea::placeholder {
    color: #777D95 !important;
}


/* ================================================
   BUTTONS
================================================ */

button.primary {
    color: white !important;

    border: none !important;
    border-radius: 12px !important;

    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1px !important;

    min-height: 49px !important;

    background: linear-gradient(
        105deg,
        #7138D8,
        #A855F7,
        #E044B5,
        #F43F8E
    ) !important;

    background-size: 200% 200% !important;

    box-shadow:
        0 6px 20px rgba(168,85,247,0.24),
        0 0 18px rgba(224,68,181,0.10);

    transition: all 0.25s ease !important;
}

button.primary:hover {
    filter: brightness(1.18);
    transform: translateY(-2px);

    box-shadow:
        0 9px 27px rgba(168,85,247,0.38),
        0 0 25px rgba(224,68,181,0.20);
}

button.primary:active {
    transform: scale(0.98);
}


/* Different button colours */

.analyze-btn button {
    background: linear-gradient(
        110deg,
        #7138D8,
        #A855F7,
        #387BFF
    ) !important;

    box-shadow: 0 6px 20px rgba(120,80,255,0.28);
}

.ask-btn button {
    background: linear-gradient(
        110deg,
        #E044B5,
        #F43F8E,
        #FF8738
    ) !important;

    box-shadow: 0 6px 20px rgba(255,61,154,0.25);
}


/* ================================================
   LABELS AND GENERAL TEXT
================================================ */

.gradio-container label {
    color: #D8D9E7 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px !important;
}

.gradio-container .prose {
    color: #E5E6F0 !important;
    font-size: 15px !important;
}

.gradio-container input,
.gradio-container textarea {
    color: white !important;
}


/* ================================================
   FOOTER
================================================ */

#app-footer {
    position: relative;
    text-align: center;

    color: #9296AC;

    padding: 28px 10px;
    margin-top: 30px;

    font-size: 12px;
    line-height: 1.9;

    border-top: 1px solid rgba(255,255,255,0.10);
}

#app-footer strong {
    color: #64DFFF;
}

#app-footer .footer-brand {
    font-family: 'Inter', sans-serif !important;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: none;

    margin-bottom: 7px;

    background: linear-gradient(
        90deg,
        #A855F7,
        #387BFF,
        #00D9FF,
        #FF3D9A,
        #FF8738
    );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

footer {
    display: none !important;
}


/* ================================================
   SCROLLBAR DESIGN
================================================ */

#caption-panel::-webkit-scrollbar,
#answer-panel::-webkit-scrollbar {
    width: 6px;
}

#caption-panel::-webkit-scrollbar-thumb {
    background: var(--gold);
    border-radius: 10px;
}

#answer-panel::-webkit-scrollbar-thumb {
    background: var(--pink);
    border-radius: 10px;
}

#caption-panel::-webkit-scrollbar-track,
#answer-panel::-webkit-scrollbar-track {
    background: #171927;
}


/* ================================================
   DESKTOP RESPONSIVENESS
================================================ */

@media screen and (min-width: 1200px) {

    .gradio-container {
        padding-left: 45px !important;
        padding-right: 45px !important;
    }

    #app-header {
        padding: 58px 25px;
    }

    .panel {
        padding: 22px !important;
    }
}


/* ================================================
   TABLET RESPONSIVENESS
================================================ */

@media screen and (max-width: 900px) {

    .gradio-container {
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    .panel {
        padding: 18px !important;
    }

    .section-title h3 {
        font-size: 24px !important;
    }
}


/* ================================================
   MOBILE RESPONSIVENESS
================================================ */

@media screen and (max-width: 700px) {

    .gradio-container {
        padding-left: 9px !important;
        padding-right: 9px !important;
    }

    #app-header {
        padding: 32px 14px;
        margin: 10px 0 18px 0;
        border-radius: 17px;
    }

    #app-header h1 {
        font-size: 38px !important;
        letter-spacing: -1.5px;
    }

    #app-header p {
        font-size: 14px;
    }

    #app-header .brand-mark {
        width: 52px;
        height: 52px;
        font-size: 26px;
    }

    #app-header .tagline {
        font-size: 9px;
        letter-spacing: 2.5px;
    }

    #app-header .subtitle {
        font-size: 11px;
        letter-spacing: 2px;
    }

    .panel {
        padding: 14px !important;
        border-radius: 15px !important;
    }

    .section-title h3 {
        font-size: 21px !important;
        letter-spacing: -0.4px !important;
    }

    .section-note {
        font-size: 12px !important;
        margin-bottom: 10px;
    }

    .question-panel {
        margin-top: 12px !important;
    }

    .image-box {
        border-radius: 11px !important;
    }

    #caption-panel {
        max-height: 220px !important;
        padding: 12px !important;
    }

    #answer-panel {
        max-height: 190px !important;
        padding: 12px !important;
    }

    button.primary {
        width: 100% !important;
        min-height: 48px !important;
    }

    #app-footer {
        font-size: 11px !important;
    }

    #app-footer .footer-brand {
        font-size: 17px !important;
    }
}

"""


# ==================================================
# 3. FIND AVAILABLE PORT
# ==================================================

def find_free_port(start_port=7860, end_port=8000):

    for port in range(start_port, end_port + 1):

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:

            try:
                s.bind(("", port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port

            except OSError:
                continue

    raise OSError(
        f"No free ports available between {start_port}-{end_port}"
    )


# ==================================================
# 4. INITIALIZE AI MODELS
# ==================================================

def initialize_models():

    try:
        print("Loading Taero AI models...")

        captioner = GeminiCaptioner(GEMINI_API_KEY)
        vqa = GeminiVQA(GEMINI_API_KEY)
        detector = YOLOv12Detector()

        print("All Taero AI models loaded successfully!")

        return captioner, detector, vqa

    except Exception as e:
        print(f"Model initialization failed: {e}")
        raise


captioner, detector, vqa = initialize_models()


# ==================================================
# 5. IMAGE ANALYSIS
# ==================================================

def process_image(image):

    if image is None:
        yield None, "Please upload an image first.", None
        return

    try:
        import cv2

        os.makedirs("temp", exist_ok=True)

        image_path = os.path.join("temp", "uploaded.jpg")

        if isinstance(image, np.ndarray):

            cv2.imwrite(
                image_path,
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

        else:
            image.save(image_path)

        yield None, "Analyzing your image...", None

        # YOLOv12 object detection
        detected_image = detector.detect(image_path)

        yield (
            detected_image,
            "Object detection completed. Generating image description...",
            image_path
        )

        # Gemini image captioning
        caption = captioner.caption(image_path)

        yield (
            detected_image,
            f"### Image Description\n\n{caption}",
            image_path
        )

    except Exception as e:

        yield None, f"Error while analyzing image: {str(e)}", None


# ==================================================
# 6. IMAGE QUESTION ANSWERING
# ==================================================

def process_question(image_path, question):

    if not image_path:
        yield "Please analyze an image first."
        return

    if not question or not question.strip():
        yield "Please enter a question about the image."
        return

    try:

        for i in range(3):
            yield "Thinking" + "." * (i + 1)
            time.sleep(0.5)

        answer = vqa.answer(image_path, question)

        if answer:
            yield f"### Answer\n\n{answer}"
        else:
            yield "No answer was generated. Please try again."

    except Exception as e:

        yield f"Error while answering your question: {str(e)}"


# ==================================================
# 7. TAERO AI USER INTERFACE
# ==================================================

with gr.Blocks(
    title="Taero AI | See More. Understand Better.",
    css=contrast_css
) as demo:

    # HEADER

    gr.HTML("""
        <div id="app-header">

            <div class="brand-mark">✦</div>

            <div class="tagline">
                NEXT-GENERATION COMPUTER VISION
            </div>

            <h1>Taero AI</h1>

            <div class="subtitle">
                SEE MORE. UNDERSTAND BETTER.
            </div>

            <p>
                Your Intelligent Visual Assistant.
                Upload an image, discover what's inside,
                and ask questions with the power of AI.
            </p>

        </div>
    """)


    # ==============================================
    # UPLOAD AND DETECTION SECTION
    # ==============================================

    with gr.Row(equal_height=True):

        # IMAGE UPLOAD

        with gr.Column(
            scale=1,
            min_width=280,
            elem_classes=["panel", "upload-panel"]
        ):

            gr.Markdown(
                "### 01 · Upload Image",
                elem_classes=["section-title"]
            )

            gr.Markdown(
                '<div class="section-note">'
                '✦ &nbsp; Choose an image and let Taero AI explore it.'
                '</div>'
            )

            input_image = gr.Image(
                label="Your Image",
                type="numpy",
                height=300,
                show_download_button=False,
                elem_classes=["image-box"]
            )

            analyze_btn = gr.Button(
                "✦  Analyze Image",
                variant="primary",
                elem_classes=["analyze-btn"]
            )


        # DETECTION RESULTS

        with gr.Column(
            scale=1,
            min_width=280,
            elem_classes=["panel", "detection-panel"]
        ):

            gr.Markdown(
                "### 02 · Detection Results",
                elem_classes=["section-title"]
            )

            gr.Markdown(
                '<div class="section-note">'
                '◈ &nbsp; Explore the objects identified by YOLOv12.'
                '</div>'
            )

            output_image = gr.Image(
                label="YOLOv12 Object Detection",
                height=300,
                show_download_button=False,
                elem_classes=["image-box"]
            )

            output_text = gr.Markdown(
                "Your image description will appear here.",
                elem_id="caption-panel"
            )

            image_path_state = gr.State(None)


    # ==============================================
    # ASK TAERO AI SECTION
    # ==============================================

    with gr.Row():

        with gr.Column(
            elem_classes=["panel", "question-panel"]
        ):

            gr.Markdown(
                "### 03 · Ask Taero AI",
                elem_classes=["section-title"]
            )

            gr.Markdown(
                '<div class="section-note">'
                '✧ &nbsp; Curious about something in your image? '
                'Ask a question and get an AI-powered answer.'
                '</div>'
            )

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Example: How many people are visible in this image?",
                lines=2,
                elem_classes=["textbox"]
            )

            ask_btn = gr.Button(
                "✦  Ask Taero AI",
                variant="primary",
                elem_classes=["ask-btn"]
            )

            answer_output = gr.Markdown(
                "Your AI answer will appear here.",
                elem_id="answer-panel"
            )


    # ==============================================
    # FOOTER
    # ==============================================

    gr.HTML("""
        <div id="app-footer">

            <div class="footer-brand">
                Taero AI
            </div>

            See More. Understand Better.

            <br>

            Powered by <strong>YOLOv12</strong> and
            <strong>Google Gemini 2.5 Flash</strong>

            <br>

            © 2026 Taero AI | Intelligent Visual Assistant

        </div>
    """)


    # ==============================================
    # BUTTON CONNECTIONS
    # ==============================================

    analyze_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[
            output_image,
            output_text,
            image_path_state
        ]
    )

    ask_btn.click(
        fn=process_question,
        inputs=[
            image_path_state,
            question_input
        ],
        outputs=[answer_output]
    )


# ==================================================
# 8. LAUNCH APPLICATION
# ==================================================

if __name__ == "__main__":

    if os.environ.get("PORT"):
        port = int(os.environ["PORT"])
    else:
        port = find_free_port()

    print(f"Launching Taero AI on port {port}")

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
