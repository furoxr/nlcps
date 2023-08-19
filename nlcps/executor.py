from dataclasses import dataclass
from typing import List
from nlcps.llm import LLM
from nlcps.analysis_chain import AnalysisChain, AnalysisResult
from nlcps.sample_bank import SampleBank
from nlcps.types import RelatedSample


@dataclass
class NlcpExecutor:
    llm: LLM

    def analysis(
        self, user_utterance: str, analysis_chain: AnalysisChain
    ) -> AnalysisResult:
        """Analysis user utterance to get entities and whether context needed.

        Args:
            user_utterance (str): User utterance
            analysis_chain (AnalysisChain): Chain returning the analysis result

        Returns:
            AnalysisResult: Including entites related to user utterance and whether context needed
        """
        return analysis_chain.run(user_utterance)

    def retrieve(
        self,
        analysis_result: AnalysisResult,
        k: int,
        sample_bank: SampleBank,
        embedding: float,
    ) -> List[RelatedSample]:
        """Retrieve related samples from sample bank.

        Args:
            analysis_result (AnalysisResult): From analysis
            k (int): Return top k related samples
            sample_bank (SampleBank): Store all samples
            embedding (float): Embedding of user utterance

        Returns:
            List[RelatedSample]: Top k related samples
        """
        raise NotImplementedError()

    def program_synthesis(
        self, user_utterance: str, related_samples: List[RelatedSample]
    ) -> str:
        """Program synthesis leveraging the LLM, user utterance and related samples.

        Args:
            user_utterance (str): User utterance
            related_samples (List[RelatedSample]): From retrieve

        Returns:
            str: Generated program
        """
        raise NotImplementedError()
