import os
import pathlib
import subprocess
from typing import Any

import openai
import streamlit as st
from pydantic import BaseModel

st.title("LLMIMP")


class Session:
    """Management for streamlit.session_state"""

    output_dir: str = "output_images"

    def __init__(self):
        if "init" not in st.session_state:
            st.session_state.init = True
            st.session_state.time = 0
            st.session_state.source_images = []
            st.session_state.messages = []
            os.makedirs(self.output_dir, exist_ok=True)

    def is_clear(self) -> bool:
        return len(st.session_state.messages) == 0

    def append(self, data: Any):
        st.session_state.messages.append(data)

    def messages(self) -> list[Any]:
        return st.session_state.messages

    def time(self) -> int:
        return st.session_state.time

    def next_tick(self):
        st.session_state.time += 1

    def images(self) -> list[str]:
        return st.session_state.source_images

    def add_image(self, image: str):
        if image in st.session_state.source_images:
            return
        st.session_state.source_images.append(image)


session = Session()


class ImageMagickCommand(BaseModel):
    description: str
    command: str
    output: str


class ChatGPT:
    def __init__(self, model_name: str):
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self._system_prompt()

    def _system_prompt(self):
        if not session.is_clear():
            return
        system_prompt = f"""
あなたは画像処理エキスパートです。
ユーザーは初め input.png を持っています。
ユーザーの指示に従って ImageMagick の convert コマンドを一つ発行してください。
以下のフォーマットで応答してください。

```
{{
"description": "<説明>",
"command": "<ImageMagickのコマンド>",
"output": "<出力画像ファイル名>"
}}
```

ユーザーはあなたのコマンドを忠実に実行することで新しい画像を出力画像を得ます。
あなたは初めの input.png に限らず、ユーザーの出力画像を中間ファイルとして再利用することができます。
"""
        session.append({"role": "system", "content": system_prompt})

    def chat(self, user_prompt: str) -> ImageMagickCommand:
        output_image = f"img-{session.time()}.png"
        session.next_tick()
        user_message = f"""
入力画像 (この画像のみが参照可能): {session.images()}
出力画像: {output_image}
ユーザーの要求: {user_prompt}
"""
        session.append({"role": "user", "content": user_message})

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                m
                for m in session.messages()
                if m["role"] in {"system", "user", "assistant"}
            ],
            response_format=ImageMagickCommand,
        )
        data: ImageMagickCommand = completion.choices[0].message.parsed  # type: ignore

        text = f"description:{data.description}, command:{data.command}, output:{data.output}"
        st.session_state.messages.append({"role": "assistant", "content": text})
        return data


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


model_name = st.text_input(label="モデル名", value="gpt-4o-mini")
client = ChatGPT(model_name)

uploaded_file = st.file_uploader("Upload an image (.png)", type=["png"])
if uploaded_file:
    input_image_path = os.path.join(session.output_dir, "input.png")
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(input_image_path, caption="アップロードされた画像")
    session.add_image("input.png")

    # chat history
    for m in session.messages()[1:]:
        if m["role"] in {"system", "user", "assistant"}:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        elif m["role"] == "image":
            st.image(m["filepath"], caption=m["filename"])

    # new conversation
    if prompt := st.chat_input("What do you want"):
        with st.chat_message("user"):
            st.markdown(prompt)
        data = client.chat(prompt)
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
            session.add_image(data.output)
            session.append(
                {
                    "role": "image",
                    "filename": data.output,
                    "filepath": str(output_path),
                }
            )
        else:
            st.error(f"コマンドの実行に失敗した")
