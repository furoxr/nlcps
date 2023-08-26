import os
import asyncio

import pytest
from qdrant_client.models import Distance, VectorParams

from nlcps.analysis_chain import AnalysisExample, AnalysisResult
from nlcps.executor import NlcpsConfig, NlcpsExecutor, nlcps_executor_factory
from nlcps.types import DSLRuleExample, DSLSyntaxExample, RetrieveExample

@pytest.fixture(scope="module")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def dsl_syntax_exampes():
    return [
        DSLSyntaxExample(
            entities=["text"],
            code='# Get the title from all slides in the presentation\ntextRanges = select_text(scope="Presentation", name="Title")',
        ),
        DSLSyntaxExample(
            entities=["text"],
            code='# Gets the textRanges matching the string "Hello" from the provided shapes.\ntextRanges = select_text(scope=shapes, text="Hello")',
        ),
        DSLSyntaxExample(
            entities=["text"],
            code='# Formats the text in textRanges to be bold, italic, have Times New Roman font, have a single underline, have font size 24, have the color teal and be Left aligned.\nformat_text(textRanges=textRanges, bold=true, fontName="Times New Roman", horizontalAlignment="Left", size=24, color="teal", italic=true, underline="Single")',
        ),
        DSLSyntaxExample(
            entities=["text"],
            code='# Many of the argument to format statements are optional. For example, this format statement makes the text bulleted and changes its color to olive.\nformat_text(textRanges=textRanges, bulleted=true, color="#808000")',
        ),
    ]


@pytest.fixture(scope="module")
def dsl_rules_exampes():
    return [
        DSLRuleExample(
            entities=["text"],
            rule="For select_text, if scope is provided it must be a either Presentation or a variable of type shapes or slides. If no scope is provided, we select the user slide selection.",
        ),
        DSLRuleExample(
            entities=["text"],
            rule="You must select or insert an entity before formatting or deleting it.",
        ),
        DSLRuleExample(
            entities=["text"],
            rule="Never use for loops, array indexing or if/else statements.",
        ),
    ]


@pytest.fixture(scope="module")
def dsl_chat_exampes():
    return [
        RetrieveExample(
            user_utterance="Change the text format to make it look like a typewriter",
            code='text = select_text()\nformat_text(textRanges=text, fontName="Courier New", size=18, bold=false, italic=false, underline="None", color="#000000", bulleted=false, horizontalAlignment="Left")',
            context=None,
            entities=["text"],
        ),
        RetrieveExample(
            user_utterance="Change the text format to make it look elegant",
            code='text = select_text()\nformat_text(textRanges=text, fontName="Times New Roman", size=18, italic=true)',
            context=None,
            entities=["text"],
        ),
    ]


@pytest.fixture(scope="module")
def analysis_examples():
    return [
        AnalysisExample(
            utterance="Make the title text on this slide red",
            analysis_result=AnalysisResult(
                entities=["text"],
                thoughts="We can select the title text and make it red without knowing the existing text properties. Therefore we do not need context.",
                need_context=False,
            ),
        ),
        AnalysisExample(
            utterance="Add text that’s a poem about the life of a high school student with emojis",
            analysis_result=AnalysisResult(
                entities=["text"],
                thoughts="We need to know whether there is existing text on the slide to add the new poem. Therefore we need context.",
                need_context=True,
            ),
        ),
    ]


@pytest.fixture(scope="module")
async def executor(
    analysis_examples, dsl_chat_exampes, dsl_rules_exampes, dsl_syntax_exampes
):
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE")
    entites = ["text", "image", "shape", "slide", "presentation"]
    context_rules = [
        "- Adding new text needs context to decide where to place the text on the current slide.",
        "- Adding an image about a given topic does not require context.",
    ]

    config = NlcpsConfig(
        openai_api_key=key,
        openai_api_base=base,
        entities=entites,
        context_rules=context_rules,
        analysis_examples=analysis_examples,
        system_instruction="The DSL we are using is for performing actions in PowerPoint, please write DSL code to fulfill the given user utterance.",
        collection_name_prefix="test",
        dsl_examples_k=5,
        dsl_rules_k=5,
        dsl_syntax_k=5,
    )
    executor = nlcps_executor_factory(config)

    # Initialize empty vectorstore
    collections = [
        executor.retrieve_chain.dsl_rules_selector.collection_name,
        executor.retrieve_chain.dsl_examples_selector.collection_name,
        DSLSyntaxExample.collection_name
        # executor.retrieve_chain.dsl_syntax_selector.collection_name,
    ]
    for collection_name in collections:
        # Delete and create empty collections
        executor.qdrant_client.recreate_collection(
            collection_name, VectorParams(size=1536, distance=Distance.COSINE)
        )

    # Add DSL syntax, rules, examples into vectorstore
    # [executor.retrieve_chain.dsl_syntax_selector.add(e) for e in dsl_syntax_exampes]
    i = [await DSLSyntaxExample.save(e) for e in dsl_syntax_exampes]
    print(i)
    [executor.retrieve_chain.dsl_rules_selector.add(e) for e in dsl_rules_exampes]
    [executor.retrieve_chain.dsl_examples_selector.add(e) for e in dsl_chat_exampes]

    return executor


async def test_executor(executor: NlcpsExecutor):
    code = await executor.program_synthesis(
        user_utterance="Make the title text on this slide red", context=""
    )
    print(f"\ncode:\n{code}")

    code = await executor.program_synthesis(
        user_utterance="Add text that’s a poem about the life of a high school student with emojis",
        context="",
    )
    print(f"\ncode:\n{code}")

    code = await executor.program_synthesis(
        user_utterance="Delete all the text on this slide.", context=""
    )
    print(f"\ncode:\n{code}")
