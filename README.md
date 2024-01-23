# llm-driven-program-control-flow
Insights into how to use LLM for Program Control Flow and Steerable Agents.

## TOC

1. Problems
2. Structured Outputs
   1. For single/multi-class Classification
   2. Corpus Taxonomy Extraction
   3. Program Control Flow
   4. Structured Query Data Retrieval
3. Function Calling
   1. The Life of a Function Call
   2. Pros and Cons of Function Calling
   3. Function Calling State-of-the-Art
   4. Recommendations for Function Calling
4. Tools and Frameworks
   1. Pydantic
   2. Langchain
   3. LangSmith
   4. Instructor
   5. Marvin
   6. HuggingFace Open LLM Leaderboard

## Problems

* Can we use LLM to steer the execution of code inside systems and applications?
* Can we use LLM to make a decision at a conditional statement in our code where, given some logic, one path or another must be chosen?
* Can we use LLM to generate reliable symbolic, discreet features from unstructured, natural language?

LLMs face steerability issues due to their propensity to generate outputs in unstructured natural language, complicating integration into executable code. For instance, when asked to provide a concrete "yes"/"no" response, an LLM might generate an irrelevant answer, or answer "yes" (or "no") correctly, but continue generating additional text, making its output hard to parse programmatically.

Also, leveraging LLM for decision-making can be compromised by hallucinations, incomplete context, or flawed reasoning, leading to unreliable outputs. For example, an LLM might summarise a technical document by focusing on the general details instead of on the technicalities. Alternatively, an LLM could perform a reasoning inference leap based on defective parametric memory leading to an incorrect generated insight. For example, an LLM predicting market trends in Q2 2020 (i.e. global lockdowns) based on parametric memory from training data from previous years.

We explore methods for improving LLM steerability and decision-making reliability to enhance Program Control Flow in software applications, through the use of mechanisms to guarantee structured outputs from LLM. We further describe how structured outputs can be used for generating structured metadata from vast corpora. Finally, we explain how structured outputs can be used for executing Program Control Flow via function calling or by simply using structured outputs as control variables.

## Structured Outputs

Structured outputs enable the translation of unstructured, natural language inputs into a format that can be readily processed by software applications, thereby enhancing the steerability and decision-making reliability of LLMs. By defining clear, schema-conformant formats such as JSON objects or enums, we can direct the execution of code and make precise decisions based on the generated outputs. This section delves into the methodologies for creating structured outputs from LLMs, illustrating their pivotal role in refining program control flow and facilitating the extraction of structured metadata from large datasets.

### Single/multi-class Classification

In this use case, you want the LLM to chose one or more categories from only a predefined set of categories. The two main challenges implementing this use case with LLM are:

1. LLM hallucinates a non-existing category.
2. LLM interprets a category in a different semantic context than what was intended (e.g. Emotional Labelling: `frustration` instead of `anger`, i.e. `anger` is the emotion, while `frustration` is a cognitive state, that is often associated with `anger`).

To overcome these challenges, a three-step solution is implemented:

1. Provide the LLM via In-Context Learning (ICL) with a data model/schema where only acceptable categories are defined as enums for a given class.
   * Include an extra category for cases where none of the other categories apply, e.g. "other".
2. Provide the LLM via ICL with a data model/schema for the output format of its response, usually some form of a JSON object.
3. Implement a data validation step to enforce its predicted category/ies comply with the class' data schema.
   * If validation fails, pass the validation error back to the LLM so the LLM retries a valid classification.

### Corpus Taxonomy Extraction

In effect, Taxonomy Extraction is analogous to structured conceptual summarization, where given a large corpus you reduce it to its most important hierarchical concepts.

In this use case, you want the LLM to generate a use-case specific taxonomy that encompasses a broad corpus (i.e. a corpus significantly larger than what can fit in the LLM's context). The two main challenges implementing this use case with LLM are:

1. Passing a segment of the corpus in a single LLM call would generate a narrow (i.e. incomplete) taxonomy of the domain.
2. Multiple calls with different segments of the corpus would produce different taxonomical concepts and hierarchies (i.e. multiple unaligned taxonomies).

To overcome this challenges, a three-step solution is implemented:

0. (Optional but recommended) Seed a starter taxonomy (i.e. a set of top-level categories).
1. Iterate over segments of the corpus implementing `Single/multi-class Classification` as above to generate relevant categories, including data validation to enforce enum categories.
2. Request the LLM to provide an additional JSON object where it can suggest new category names that it considers are missing from the enum categories, as an unvalidated array of strings.
3. Pass the enum class with its categories to a second LLM, along with the unstructured new categories suggested by the LLM in the last step, and request it to:
   1. Insert the new categories in the most relevant place in the hierarchy.
   2. Review if the hierarchy needs to be refactored, given the new insertions.

The outputs of this process are two-fold:

1. A domain taxonomy.
2. The source corpus annotated with the taxonomy's entities.
   * Might need to be done using a second pass over the corpus, after the taxonomy has been completed in the first pass.

### Program Control Flow

In this use case, you want the LLM to replace an `if else` or `switch` statement for a control variable representing some form of unstructured natural language, e.g. "Is the user utterance a Natural Language Query (NLQ)?" (i.e. `boolean`), "Is the user question about product A, B, C, or D?" (i.e. `case`), etc.

This use case is implemented exactly like a `Single/multi-class Classification`, except that the output is not a classification category, but the input argument for a downstream caller method which uses it as a control variable within its internal control flow.

### Structured Query Data Retrieval

In the above `Program Control Flow` use case, the LLM behaves as a liaison between two segments of imperative code. In this use case, the LLM behaves as the liaison between an upstream segment of imperative code and a downstream segment of declarative code. Instead of generating an output control variable, the LLM generates a query that can be run in a declarative language, such as SQL, MongoDB, SPARQL, etc.

The two main challenges implementing this use case with LLM are:

1. The LLM must have enough reasoning ability to distinguish between a normal user utterance and a user utterance that represents a NLQ relevant to some underlying system data structure (e.g. a postgres database).
2. The LLM must have enough coding ability to generate a structured query in the relevant declarative language (e.g. postgreSQL) given the underlying system data structure (e.g. a postgres database).

For example, a user can utter the following statement in a Business Intelligence chatbot:

> Find chats where high Net Promoter Scores (NPS) correlate with specific user journey touchpoints.

In step 1, the chatbot needs to generate a boolean output control variable deciding if this qualifies as a NLQ or not. In this example, it outputs `True`, so the execution continues to step 2, where the LLM translates the NLQ to:

```sql
SELECT dp.*, ct.touchpoint_name, dp.net_promoter_score 
FROM DATA_PRODUCTS dp 
JOIN CONVERSATION_TOUCHPOINTS ct ON dp.cs_chat_id = ct.cs_chat_id 
WHERE dp.net_promoter_score >= 8;
```

A downstream system that executes declarative code consumes the above query and retrieves the relevant data.

These two steps don't have to be implemented by the same LLM. Often the first step can be executed by a less capable LLM (e.g. `GPT-3.5-turbo`) than the one executing the second step (e.g. `GPT-4`), as the former only requires generating a boolean output, while the latter must generate valid executable code. Using the index of [LLM benchmarks](https://github.com/leobeeson/llm_benchmarks) and HuggingFace's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), once can find an LLM that scores high in a `Knowledge and Language Understanding` or `Reasoning Capabilities` benchmark for step 1, and another LLM that scores high in a `Coding Capabilities` benchmark for step 2.

## Function Calling

Function calling is a special case of Structured Outputs generated by LLM. Specifically, function calling is a mechanism for prompting an LLM so it generates a JSON object output which it can unpack as the input arguments of a function which the LLM client (on the client side) can directly call.

### The Life of a Function Call

Although function calling can have subtle differences between LLM clients, this is usually the general procedure:

1. In the prompt, the developer provides the LLM with one or several function names and corresponding parameter definitions as "tools" it can invoke when answering the user's query.
2. When trying to answer the user's query, the LLM decides (or can be forced, depending on the LLM client) if it needs to invoke one of the functions provided as "tools".
3. If it choses to invoke one, it first generates a structured output representing the arguments for the parameters of the function it wants to invoke.
4. At this point, the LLM output is halted (often with the use of a special stop sequence), and the invoked function is called client-side with the arguments generated by the LLM.
5. The output generated by the function is now passed back to the halted LLM call (server-side).
6. The LLM now uses the function's output to complement its parametric memory and generate an answer to the user's query.

### Pros and Cons of Function Calling

Function calling are a powerful instrument for abstracting away imperative code which a developer would usually have to write into the program. If a specific application integrates with a dozen or more other services, programming said integration could be extremely complex. With function calling, the developer can provide the LLM with only the functions, and the LLM takes care of the logic of when and how to call which one and in which order. This is why function calling is particularly attractive for systems that require accessing external tools and APIs.

However, there are several challenges and risks associated with using function calls in applications. The main three challenges/risks are:

1. You must include detailed exception handling in your function definitions and reiterate across the LLM prompt instructions on how to handle them.
2. Many LLM responses are non deterministic, making exception handling more cumbersome than in normal imperative code.
3. When using third-party LLM, a model update can disrupt how your prompt instructs the LLM to handle an error or exception.

### Function Calling State-of-the-Art

Function calling is a very novel mechanism, which very few LLM can handle, and even fewer can do so properly. This is because to properly implement function calling, an LLM must have been trained with function calling data in order to learn the halt-pass-receive-generate pattern. Currently, function calling is possible successfully with the [`GPT-4` and `GPT-3.5-turbo` models from OpenAI](https://platform.openai.com/docs/guides/function-calling). OpenAI models can even perform [parallel function calling](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling) enabling the LLM with the ability to perform multiple function calls together, allowing the effects and results of these function calls to be resolved in parallel.

Anthropic has [experimental capabilities for function calling for `Claude-2.1`](https://docs.anthropic.com/claude/docs/claude-2p1-guide#experimental-tool-use), but it appears it can particularly [hallucinate during function calls](https://docs.anthropic.com/claude/docs/claude-2p1-guide#hallucination-mitigation), and writing a prompt with function calling that prevents hallucination is currently very cumbersome as in [this example](https://docs.google.com/spreadsheets/d/1sUrBWO0u1-ZuQ8m5gt3-1N5PLR6r__UsRsB7WeySDQA/edit#gid=464806744). Finally, while function calls with OpenAI models can be comfortably defined with JSON schemas, `Claude-2.1` has ~~for some bizarre reason~~ been trained to perform function calling with xml tags.

A new Open Source model that has been specifically trained to perform function calls is [Gorilla](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html) from UC Berkeley, which according to its benchmarks, it only slightly trails behind OpenAI's models.

### Recommendations for Function Calling

Two questions that arise now are:

1. Should we use function calling or simply use other structured output patterns and handle dependant functions externally to the LLM call?
2. If function calling does offer some benefits in certain scenarios, when should we use them?

As with much of the interactions with LLM in the present, it is often more an art than a science. As a rule of thumb, use function calling with functions with minimal number of parameters, preferably one, maximum two. If passing multiple functions to an LLM as a set of tools from which to choose, make sure each function is orthogonal to each other with regards to the type of decision the LLM needs to do to decide which one to choose. If two or more functions serve conceptually similar purposes, better to handle each in a different LLM request.

At this time, even by following the above tips, it is highly probable many of your function calls will fail, especially if you're not using `GPT-4`. At this time, it is safer to implement generation of Structured Outputs as mentioned above, and call the functions imperatively. However, function calling is very likely a capability that will be significantly advanced in the very near future, so worth knowing how to implement and benchmark them.

## Tools and Frameworks

The number of tools and frameworks that have appeared in the last 12 months is extraordinary. Paradoxically, with every new model update by OpenAI and with the publishing of many new Open Source LLM, many of these tools and frameworks become obsolete instantaneously. However, a few tools continue being relevant and are worth mastering for working with LLM beyond the amateur use case of unstructured text generation.

### Pydantic

If you can only learn one tool for working with LLM, learn how to be proficient with `Pydantic`. `Pydantic` is a popular data validation library for Python, known for its efficiency and flexibility. Its primary strength lies in using Python type hints for schema validation and serialization. This design aligns well with modern Python coding practices, making it intuitive to use while ensuring strong integration with IDEs and static analysis tools. `Pydantic`'s core validation logic is written in Rust, contributing to its status as one of the fastest data validation libraries in the Python ecosystem. It supports a wide range of Python's standard library types and enables the creation of custom validators and serializers. Additionally, `Pydantic` can emit JSON Schema, facilitating easy integration with other tools and systems [source](https://docs.pydantic.dev/latest/).

In the context of LLM, `Pydantic`'s ability to define structured output schemas is particularly useful. `Langchain`, for example, utilizes `Pydantic` in its Output Parsers to convert the text generated by LLMs into structured data. This capability is crucial when building applications that require processing and integrating LLM outputs into existing models or databases. For instance, `Langchain`'s `Pydantic` (JSON) Parser allows users to specify an arbitrary JSON schema and query LLMs for outputs that conform to this schema. This structured approach to handling LLM outputs not only ensures more reliable data handling but also enhances the model's practical utility in various applications [source](https://python.`Langchain`.com/docs/modules/model_io/output_parsers/`Pydantic`) [source](https://www.gettingstarted.ai/how-to-`Langchain`-output-parsers-convert-text-to-objects/).

### Langchain

`Langchain` is a powerful framework for building language model applications. It provides modules that can be used to create complex applications by combining different components. The core component of `Langchain` is the Language Model (LLM), which can be either an LLM or a ChatModel. The LLM is the reasoning engine that takes input and generates output. `Langchain` also includes Prompt Templates, which provide instructions to the LLM, and Output Parsers, which translate the raw LLM output into a more usable format. [Langchain Docs](https://python.langchain.com/docs/get_started/quickstart)

`Langchain` improves working with LLMs in several ways:

* **Task Decomposition:** `Langchain` allows breaking down complex tasks into smaller subtasks, making it easier to handle them with an LLM or other components of the agent system.
* **LLM as Core Controller:** `Langchain` uses the LLM as the primary controller of an autonomous agent system. The LLM is complemented by other key components such as a knowledge graph and a planner.
* **Potential of LLM:** `Langchain` recognizes the potential of LLMs as powerful general problem solvers. LLMs can be used not only for generating well-written copies but also for solving complex tasks and achieving human-like intelligence.
* **Challenges in Long-Term Planning:** `Langchain` acknowledges the challenges in planning over a lengthy history and effectively exploring the solution space. These challenges are important limitations of current LLM-based autonomous agent systems.

### LangSmith

`Langsmith` is a unified platform designed to enhance the development, debugging, testing, and monitoring of language model applications. It is built on top of `Langchain`, an open-source framework for building applications using large language models (LLMs). `Langsmith` provides a user-friendly interface and a set of powerful tools that streamline the development process and improve the overall workflow. [Langsmith Docs](https://docs.smith.langchain.com/)

`Langsmith` improves working with LLMs in several ways:

* **Efficient Development:** `Langsmith` simplifies the development process by providing a range of features and functionalities specifically tailored for working with LLMs. It offers prompt templates, which provide instructions to the LLM, and output parsers, which help translate the raw LLM output into a more usable format.
* **Debugging and Testing:** `Langsmith` includes debugging and testing tools that allow developers to identify and fix issues in their language model applications. It provides capabilities for step-by-step debugging, error handling, and testing different scenarios to ensure the robustness and reliability of the applications.
* **Monitoring and Optimization:** `Langsmith` enables developers to monitor the performance of their language model applications in real-time. It provides insights into the model's behavior, performance metrics, and resource utilization, allowing developers to optimize and fine-tune their applications for better efficiency and effectiveness.
* **Collaboration and Version Control:** `Langsmith` offers collaboration features that facilitate teamwork and version control. It allows multiple developers to work on the same project simultaneously, track changes, and manage different versions of the application, ensuring seamless collaboration and efficient project management.

### Instructor

If using OpenAI models, `Instructor` is a highly recommended framework. It is a Python library designed for structured data extraction, leveraging OpenAI's function calling API and `Pydantic`. It stands out for its simplicity, transparency, and control, making it intuitive. Instructor utilizes Pydantic for schema validation and type hints, offering customization through validators and integration with IDEs. The library supports CLI tools for tracking OpenAI usage and offers a large collection of very useful examples in its cookbook.

Recommended resources for getting started with `Instructor`:

* [Pydantic is all you need: Jason Liu](https://www.youtube.com/watch?v=yj-wSRJwrrc)
* [Instructor Homepage](https://jxnl.github.io/instructor/).
* [Instructor GitHub](https://github.com/jxnl/instructor)

### Marvin AI

Marvin is a lightweight AI engineering framework designed for building natural language interfaces. It emphasizes best practices in software development, bringing these to generative AI. Marvin offers core components like AI Models, AI Classifiers, AI Functions, and AI Applications to structure text into schemas, perform classification and routing, handle complex logic, and manage interactive use. It aims to deliver Ambient AI, integrating AI seamlessly into software stacks for making unstructured data universally accessible. Marvin is created by the team behind [Prefect](https://www.prefect.io/), leveraging their experience in open-source developer tools.

* [Marvin Homepage](https://www.askmarvin.ai/welcome/what_is_marvin/).
* [Marvin GitHub](https://github.com/PrefectHQ/marvin)

### HuggingFace Open LLM Leaderboard

HuggingFace Open LLM Leaderboard is an important resource for searching for Open Source LLM based on their capabilities as measured by their evaluations against LLM benchmarks. You can filter LLM by benchmarks, model types, model size, etc. Once you have a clearly specified use case for your application, you can search for the LLM that best suits said use case:

* [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
