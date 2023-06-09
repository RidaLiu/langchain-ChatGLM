from abc import ABC
import os
import requests
from typing import Optional, List
from langchain.llms.base import LLM

from models.loader import LoaderCheckPoint
from models.base import RemoteRpcModel, AnswerResult
from typing import Collection, Dict


def _build_message_template() -> Dict[str, str]:
    """
    :return: 结构
    """
    return {
        "role": "",
        "content": "",
    }


def _read_file(file_path: str) -> str:
    if os.path.exists(file_path):
        r = open(file_path, "r")
        return r.read()
    return ""


class FastChatOpenAILLM(RemoteRpcModel, LLM, ABC):
    api_base_url: str = "http://localhost:8000/v1"
    model_name: str = "chatglm-6b"
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "FastChat"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    @property
    def _api_key(self) -> str:
        pass

    @property
    def _api_base_url(self) -> str:
        return self.api_base_url

    def set_api_key(self, api_key: str):
        pass

    def set_api_base_url(self, api_base_url: str):
        self.api_base_url = api_base_url

    def call_model_name(self, model_name):
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    # 将历史对话数组转换为文本格式
    def build_message_list(self, query) -> Collection[Dict[str, str]]:
        build_message_list: Collection[Dict[str, str]] = []
        history = (
            self.history[-self.history_len :] if self.history_len > 0 else []
        )
        for i, (old_query, response) in enumerate(history):
            user_build_message = _build_message_template()
            user_build_message["role"] = "user"
            user_build_message["content"] = old_query
            system_build_message = _build_message_template()
            system_build_message["role"] = "system"
            system_build_message["content"] = response
            build_message_list.append(user_build_message)
            build_message_list.append(system_build_message)

        user_build_message = _build_message_template()
        user_build_message["role"] = "user"
        user_build_message["content"] = query
        build_message_list.append(user_build_message)
        return build_message_list

    def generatorAnswer(
        self,
        prompt: str,
        history: List[List[str]] = [],
        streaming: bool = False,
    ):
        try:
            import openai

            # Not support yet
            if os.getenv("CALL_AZURE_OPENAI") == "True":
                openai.api_type = "azure"
                openai.api_base = _read_file(
                    os.getenv("AZURE_OPENAI_ENDPOINT_FILE")
                )
                openai.api_version = _read_file(
                    os.getenv("AZURE_OPENAI_VERSION_FILE")
                )
                openai.api_key = _read_file(os.getenv("AZURE_OPENAI_KEY_FILE"))
                completion = openai.ChatCompletion.create(
                    engine=_read_file(os.getenv("AZURE_OPENAI_ENGINE_FILE")),
                    model=self.model_name,
                    messages=self.build_message_list(prompt),
                )
            else:
                openai.api_key = "EMPTY"
                openai.api_base = self.api_base_url
                # create a chat completion
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=self.build_message_list(prompt),
                )
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        history += [[prompt, completion.choices[0].message.content]]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {
            "answer": completion.choices[0].message.content
        }

        yield answer_result
