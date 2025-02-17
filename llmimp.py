import base64
import hashlib
import mimetypes
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

import openai
import PIL.Image
import streamlit as st
from pydantic import BaseModel

st.title("LLMIMP")
st.subheader("v2025.02.17.alpha")


def tobase64(path: str) -> str:
    img_type, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        img_b64_str = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{img_type};base64,{img_b64_str}"


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "image"]
    content: Any | None = None
    content_for_user: Any | None = None
    filename: str | None = None
    filepath: str | None = None


class Messages:
    system: Message | None
    system_image: Message | None
    chat: list[tuple[Message, Message, Message]]  # user, assistant, image
    chat_length: int

    def __init__(self):
        self.system = None
        self.system_image = None
        self.chat = []
        self.chat_length = 0

    def clear(self):
        self.chat.clear()

    def undo(self):
        if self.chat_length > 0:
            self.chat_length -= 1

    def redo(self):
        if self.chat_length + 1 <= len(self.chat):
            self.chat_length += 1

    def trim(self):
        self.chat = self.chat[: self.chat_length]

    def iter(self, visual_mode: bool):
        if self.system:
            yield self.system
        if self.system_image and visual_mode:
            yield self.system_image
        for x, y, z in self.chat[: self.chat_length]:
            yield x
            yield y
            yield z

    def append(self, m_user: Message, m_assistant: Message, m_image: Message):
        self.trim()
        self.chat_length += 1
        self.chat.append((m_user, m_assistant, m_image))


class Session:
    """Management for streamlit.session_state"""

    output_dir: str = "output_images"

    def __init__(self):
        if "init" not in st.session_state:
            st.session_state.init = True
            st.session_state.messages = Messages()
            st.session_state.input_shown = False
            os.makedirs(self.output_dir, exist_ok=True)

    def time(self) -> int:
        return st.session_state.messages.chat_length

    @property
    def messages(self):
        return st.session_state.messages

    def images(self) -> list[str]:
        return ["input.png"] + [
            m.filename for m in self.messages.iter(False) if m.role == "image"
        ]


session = Session()


class ImageMagickCommand(BaseModel):
    description: str
    command: str
    output: str


class ChatGPT:
    def __init__(self, model_name: str, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self._system_prompt()

    def _system_prompt(self):
        session.messages.system = Message(
            role="system",
            content="""
あなたは画像処理エキスパートです。
ユーザーは初め input.png を持っています。
ユーザーの指示に従って ImageMagick の convert コマンドを一つ発行してください。
以下のフォーマットで応答してください。

```
{
"description": "<説明>",
"command": "<ImageMagickのコマンド>",
"output": "<出力画像ファイル名>"
}
```

ユーザーはあなたのコマンドを忠実に実行することで新しい画像を出力画像を得ます。
あなたは初めの input.png に限らず、ユーザーの出力画像を中間ファイルとして再利用することができます。
""",
        )

    def show_visual(self, image_path: str):
        image_url = tobase64(image_path)
        session.messages.system_image = Message(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "こちらが実際の input.png です. 参考にして",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        )

    def chat(
        self, user_prompt: str, visual_mode: bool
    ) -> tuple[ImageMagickCommand, Message, Message]:
        output_image = f"img-{session.time()}.png"
        m_user = Message(
            role="user",
            content=f"""
入力画像 (この画像のみが参照可能): {session.images()}
出力画像: {output_image}
ユーザーの要求: {user_prompt}
""",
            content_for_user=user_prompt,
        )
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": m.role, "content": m.content}
                for m in list(session.messages.iter(visual_mode)) + [m_user]
                if m.role in {"system", "user", "assistant"}
            ],
            response_format=ImageMagickCommand,
        )
        data: ImageMagickCommand = completion.choices[0].message.parsed  # type: ignore
        text = f"description:{data.description}, command:{data.command}, output:{data.output}"
        json = {
            "description": data.description,
            "command": data.command,
            "output": data.output,
        }
        m_assistant = Message(role="assistant", content=text, content_for_user=json)
        return data, m_user, m_assistant


class ImageMagick:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, command: str):
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.output_dir, capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"コマンドの実行中にエラーが発生しました: {result.stderr}")
                return False
            else:
                return True
        except Exception as e:
            st.error(f"コマンドの実行中に例外が発生しました: {e}")
            return False


# API KEY
api_key = None
with st.sidebar:
    if os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = st.text_input("OPENAI_API_KEY", key="openai_api_key")

        # If you have the password, you can use master API KEY
        master_api_key = str(st.secrets.get("openai_api_key"))
        password = st.secrets.get("password")
        hashed = hashlib.md5((api_key + master_api_key).encode("utf-8")).hexdigest()
        if hashed == password:
            api_key = master_api_key
            st.info("Master KEY used")

if not api_key:
    st.error("Open the sidebar (←) and enter your OPENAI_API_KEY")
    st.stop()

model_name = st.selectbox(label="モデル名", options=["gpt-4o-mini", "gpt-4o", "o1"])

client = ChatGPT(model_name, api_key)
visual_mode = st.checkbox("Visual mode", value=True, help="オフにすると画像を見ないでコマンドを生成する")

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png", "gif"])
if uploaded_file:
    input_image_path = os.path.join(session.output_dir, "input.png")
    image = PIL.Image.open(uploaded_file)

    # Image size
    width, height = image.size
    maxwidth = int(
        st.slider("width", min_value=10, max_value=width, value=width, step=10)
    )
    maxheight = int(
        st.slider("height", min_value=10, max_value=height, value=height, step=10)
    )

    # shrink
    if maxwidth < width or maxheight < height:
        image.thumbnail((maxwidth, maxheight))

    image.save(input_image_path, format="PNG")
    st.image(input_image_path, caption="アップロードされた画像 (input.png)")

    if visual_mode:
        client.show_visual(input_image_path)

    # chat history
    for m in session.messages.iter(visual_mode):
        if m.role in {"user", "assistant"} and m.content_for_user:
            with st.chat_message(m.role):
                st.write(m.content_for_user)
        elif m.role == "image":
            st.image(m.filepath, caption=m.filename)
            with open(m.filepath, "rb") as imagefile:
                st.download_button(
                    ":material/download:",
                    data=imagefile,
                    file_name=m.filename,
                    mime=mimetypes.guess_type(m.filepath)[0],
                )

    # new conversation
    if prompt := st.chat_input("What do you want?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        data, m_user, m_assistant = client.chat(prompt, visual_mode)
        with st.chat_message("assistant"):
            st.json(
                {
                    "description": data.description,
                    "command": data.command,
                    "output": data.output,
                }
            )

        success = ImageMagick(session.output_dir).run(data.command)
        output_path = pathlib.Path(session.output_dir) / data.output
        if success and output_path.exists():
            st.image(str(output_path), caption=data.output)
            with open(str(output_path), "rb") as imagefile:
                st.download_button(
                    ":material/download:",
                    data=imagefile,
                    file_name=data.output,
                    mime=mimetypes.guess_type(data.output)[0],
                )
            m_image = Message(
                role="image",
                filename=data.output,
                filepath=str(output_path),
            )
            session.messages.append(m_user, m_assistant, m_image)

        else:
            st.error("コマンドの実行に失敗した")

    with st.container():
        left, mid, right = st.columns(3)
        with left:
            st.button(
                ":material/undo: Undo",
                use_container_width=True,
                on_click=session.messages.undo,
            )
        with mid:
            st.button(
                ":material/redo: Redo",
                use_container_width=True,
                on_click=session.messages.redo,
            )
        with right:
            st.button(
                ":material/clear: All clear",
                use_container_width=True,
                on_click=session.messages.clear,
            )

    # with st.sidebar:
    #     with st.expander("DEBUG"):
    #         st.write(
    #             {
    #                 "time": session.time(),
    #                 "chat_length": session.messages.chat_length,
    #                 "len(chat)": len(session.messages.chat),
    #             }
    #         )
