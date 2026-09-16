
AVG Architecture Brief: Path 2 — Structural Code Exclusion FilterDoc ID: AVG-RFC-002Status: PROPOSED / DEFERREDTarget Release: v2.2-code-filter-experimentAuthors: Active Variety Governor Research Group

1. Problem Statement & Theoretical MotivationUnder 1D and 2D scalar metric governance (token_diversity and trailing_ctr), a fundamental boundary ambiguity exists in multi-pretrained base models like Qwen2.5-1.5B.When a model generates code comment delimiters, docstrings, or structural syntax (e.g., //using \n //array, # %%\n, /* ... */), the localized character distribution exhibits:Low Coherent Token Ratio ($\text{CTR} < 0.30$): Due to short symbols, single letters, and non-alphabetic operators.Transient Diversity Dips ($\text{Dist-2} < 0.30$): Due to repeated structural tokens like //, \n, or ->.In 2D scalar space, this structural code signature overlaps with genuine text repetition traps (word word word...). Scalar thresholding cannot separate these two distinct generative manifolds without either:Sacrificing Factual Safety: Loosening thresholds to catch Word Loop causes false-positive interventions on code comment blocks.Sacrificing Residual Coverage (Path 1): Tightening thresholds via AND conjunction guarantees $100\%$ factual safety, but forces Word Loop exits to rely entirely on logit-side controls rather than residual steering.To break this scalar trade-off, Path 2 introduces a 3rd feature dimension: Lexical Code Syntax Density.

2. Proposed Path 2 ArchitectureInstead of relying solely on floating-point metrics, Path 2 adds a Lightweight Structural Code Classifier to governor/controller.py.
```
                        ┌───────────────────────────────────────┐
                        │ Trailing Ring Buffer (Last 16 Tokens)  │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │   3D Multi-Modal Diagnostic Evaluation           │
                  ├────────────────────────┬─────────────────────────┤
                  │ 1. Token Diversity     │ compute_distinct_2()    │
                  │ 2. Trailing CTR        │ compute_ctr()           │
                  │ 3. Code Syntax Density │ detect_code_syntax()    │
                  └────────────────────────┴──────────┬──────────────┘
                                                      │
                                                      ▼
                                       Is Code Syntax Marker Present?
                                          /                       \
                                        YES                        NO
                                        /                           \
                           ┌────────────────────────┐   ┌────────────────────────┐
                           │ FORCE is_token_loop=0  │   │ Evaluate Standard OR   │
                           │ Suppress Residual Reset│   │ Predicate for Residual │
                           │ (Maintain 100% Safety) │   │ Steering Exit          │
                           └────────────────────────┘   └────────────────────────┘
                           ```
3. Lexical Code Detector Implementation SpecificationThe detector analyzes the trailing 16-token text window for explicit code formatting tokens:
```Python
import re

# Defined Code Syntax Patterns
CODE_DELIMITER_PATTERNS = [
    r'//',          # C/C++/Java/JS single-line comments
    r'/\*',         # Multi-line comment open
    r'#\s*%%',      # Jupyter / VSCode code cell breaks
    r'\bdef\s+',     # Python function definition
    r'\blet\s+',     # JS variable declaration
    r'\bconst\s+',   # JS constant declaration
    r'\bvar\s+',     # JS/Go variable declaration
    r'\bstruct\b',   # C/C++/Rust struct keyword
    r'\btypedef\b',  # C/C++ typedef keyword
    r'```',          # Markdown code fences
]

CODE_REGEX = re.compile('|'.join(CODE_DELIMITER_PATTERNS))

def is_code_syntax_context(text: str) -> bool:
    """
    Returns True if the trailing window contains structural code indicators,
    allowing the governor to bypass false-positive residual resets.
    """
    if not text:
        return False
    return bool(CODE_REGEX.search(text))
    ```

4. Integrated Diagnostic Gate Logic (Path 2)Under Path 2, diagnose() in governor/controller.py will be modified as follows:
```Python
    def diagnose(
        self,
        profile: VarietyProfile,
        input_ids: Optional[torch.Tensor] = None,
    ) -> List[InterventionDecision]:
        decisions: List[InterventionDecision] = []
        if not self.baseline.is_calibrated():
            return decisions

        mid_start = int(self._n_layers * 0.55)
        token_diversity = 1.0
        trailing_ctr = 1.0
        is_code_context = False

        if input_ids is not None:
            token_diversity, _ = compute_token_distinct_2(input_ids)
            recent_tokens = input_ids[0, -16:]
            if self.tokenizer is not None and recent_tokens.numel() > 0:
                recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
                trailing_ctr = compute_coherent_token_ratio(recent_text)
                # Evaluates explicit structural syntax density
                is_code_context = is_code_syntax_context(recent_text)

        # 1. DUAL-GATE IMMUNITY & STATE RESET
        if trailing_ctr >= 0.75 and token_diversity >= 0.35:
            self._consecutive_interventions = {}
            self._collapse_persistence_counter = 0
            return decisions

        # 2. PATH 2 STRUCTURAL CODE EXCLUSION
        # If text is inside an active code syntax block, disable residual resets
        # to prevent false positives on comment headers or struct syntax.
        if is_code_context:
            self._collapse_persistence_counter = 0
            self._consecutive_interventions = {}
            return decisions

        # 3. EXPANDED OR PREDICATE FOR NATURAL TEXT LOOPS
        # Re-activates broader residual steering for natural text loops (div < 0.25 OR ctr < 0.20)
        is_token_loop = token_diversity < 0.25 or trailing_ctr < 0.20

        if is_token_loop:
            self._collapse_persistence_counter += 1
        else:
            self._collapse_persistence_counter = 0
            self._consecutive_interventions = {}
            return decisions

        # 4. PERSISTENCE HYSTERESIS GATE
        if self._collapse_persistence_counter < 2:
            return decisions

        # [Proceed with single deepest commitment layer selection and L2 impulse application...]
```
5. Hypothesized Gains & Risk Matrix for Path 2Metric / DomainExpected Behavior in Path 2Risk / Verification RequirementFactual Safety ("Fox" Prompt)$100\%$ Immunity Maintained//using \n //array will trigger is_code_context = True, forcing immediate governor dormancy.Word Loop Residual RescueRestoredPure word loops ("word word...") contain no code delimiters, allowing div < 0.25 to trigger residual steering.Symbol SpamRestoredPure symbol traps ("# % # %...") without keywords will trigger residual steering.Edge Case RiskCode Repetition LoopsIf the model enters an infinite loop of raw code keywords (e.g., def def def def), is_code_context could suppress residual action. (Mitigated by logit-side $N$-gram penalties).

6. Execution Criteria for Initiating Path 2 TestingPath 2 should be un-shelved and implemented when:Benchmarking models trained specifically on code/math corpora (e.g., StarCoder, DeepSeek-Coder, CodeLlama) where code-syntax false positives dominate.Expanding the test suite to evaluate residual vector steering quality on complex multi-word repetition loops.Conducting ablation studies comparing pure logit-side kickstart exits vs. joint logit + residual steering exits.
