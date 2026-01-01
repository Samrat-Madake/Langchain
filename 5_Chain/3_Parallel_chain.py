from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # For Parsing output into String

from langchain_core.runnables import RunnableParallel # For Parallel Chain

from dotenv import load_dotenv
load_dotenv()

#  LLM 1 
llm1 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

#  LLM 2 
llm2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

# #  Model 2 
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     # repo_id="google/gemma-2-2b-it",
#     task="text-generation",
#     temperature=0.7
# )
# model = ChatHuggingFace(llm=llm)

# Prompt Template 1
template1 = PromptTemplate(
    template=
        "Generate a Short simple notes on given text {text}.",
    input_variables=["text"],
    )

# Prompt Template 2
template2 = PromptTemplate(
    template=
        "Generate 5 Short Question Answers from following text {text}.",
    input_variables=["text"],
    )

# Prompt Template 3 :  merge 
template3 = PromptTemplate(
    template=
        "Merge the provided notes and Quiz into a single document {notes} and {quiz}.",
    input_variables=["notes", "quiz"],
    )

#  llm : Grok, model : Huggingface
# Parser
parser = StrOutputParser()

# Parallel Chain
parallel_chain = RunnableParallel({
    "notes": template1 | llm1 | parser,
    "quiz": template2 | llm2 | parser,
})

# Merge Chain
merge_chain = template3 | llm1 | parser

# Full Chain
chain = parallel_chain | merge_chain

#  Topic : Semiconductor Industry Report
text ='''
Detailed report — The Semiconductor Industry (≈2,000–3,000 words)

Executive summary
Semiconductors are the foundational building blocks of modern electronics — tiny solid-state devices that control current flow and enable computation, sensing and communication. Over the past 80 years the industry has evolved from discrete transistors to trillions of transistors per chip, driven by continuous innovation in device physics, materials, lithography and packaging. Today the market is large and growing (hundreds of billions of dollars annually), driven by megatrends: artificial intelligence (AI), cloud/datacenters, mobile computing, automotive electrification/autonomy, and ubiquitous sensing (IoT). At the same time the industry faces technology limits (cost and complexity of sub-nm scaling), supply-chain & geopolitics pressures, and a transition from monolithic scaling to heterogeneous integration (chiplets, advanced packaging). This report explains the technical foundations, the manufacturing chain, business dynamics, current issues and likely near-term future directions — with concrete examples to make concepts tangible.

1. What is a semiconductor? (fundamentals, with examples)

A semiconductor is a material whose electrical conductivity lies between that of a conductor and an insulator; silicon is the dominant semiconductor material. Semiconductors are engineered into devices (diodes, transistors) by doping (adding controlled impurities), building dielectric layers, and patterning metal interconnects. The most important device family for modern logic and memory is the MOSFET (metal-oxide-semiconductor field-effect transistor).

Example: a smartphone’s application processor uses billions of CMOS transistors (complementary MOS) to execute code, while flash memory uses floating-gate structures to store bits non-volatilely.

2. Short history and the meaning of “node” and Moore’s Law (with examples)

Key milestones: the point-contact transistor (1947), the silicon integrated circuit (late 1950s), planar MOS processes in the 1960s, and the explosive growth of transistor packing summarized by Moore’s Law — the empirical observation that transistor density roughly doubles every ~18–24 months. A “process node” (e.g., 7 nm, 5 nm, 3 nm) historically referred to the minimum feature printed on chip layers; today it’s more a marketing / capability label tied to transistor performance, density and power than a literal single dimension.

Example: Early microprocessors in the 1970s had thousands of transistors; contemporary CPUs and GPUs have tens of billions — enabling AI models that require massive parallel compute.

(Background reading: timelines & Moore’s law discussions). 
CHM
+1

3. How chips are made — the manufacturing chain (simplified step-by-step)

Design (front end) — architects design logic, memory, I/O and produce a GDSII/OSP file (layout). Example tools: Cadence, Synopsys.

Mask/reticle creation — high-resolution patterns are written for lithography.

Wafer fab (front end of line) — silicon wafers undergo oxidation, deposition, ion implantation (doping), etching, and repeated lithography + etch cycles to build devices layer by layer. Lithography is the critical enabler of feature size reduction.

Back end of line (BEOL)/metallization — multiple metal layers and vias interconnect transistors into circuits.

Testing and packaging — wafer probe tests die; good dies are singulated, packaged (BGA, flip-chip, etc.), and tested again. Packaging now plays a strategic role (see later).

Distribution & system integration — chips are assembled into boards and systems.

The most technology-intensive, capital-heavy part is the wafer fab: modern fabs cost many billions of dollars and require complex equipment (lithography scanners, deposition systems, metrology).

4. Lithography and why EUV matters (short technical note)

Lithography prints circuit patterns on photoresist using light. As features shrink, conventional deep-UV (DUV) lithography needed complex multipatterning. Extreme Ultraviolet (EUV) lithography (13.5 nm wavelength) enabled simpler patterning for advanced nodes (7/5/3 nm). ASML (Netherlands) is the dominant supplier of EUV scanners and therefore a pivotal supplier in the ecosystem. Recent advances (High-NA EUV) promise further resolution gains but are complex and expensive to deploy. 
ASML
+1

Example: moving from 7 nm to 3 nm required extensive EUV use; chipmakers that could access EUV achieved competitive density and performance advantages.

5. Who does what — the industry structure and leading players

The industry has specialized players across the value chain:

IDMs (Integrated Device Manufacturers) — design and manufacture internally (e.g., Intel historically).

Foundries — manufacture for fabless customers (TSMC, Samsung Foundry). Foundries enable many fabless startups and system-level companies (e.g., Apple uses TSMC for its A-series/ M-series chips).

Fabless designers — design chips but outsource manufacturing (NVIDIA, AMD, Qualcomm, Broadcom).

EDA & IP providers — design automation and reusable blocks (Cadence, Synopsys, Arm).

Equipment makers & materials suppliers — ASML (lithography), Applied Materials, Lam Research, KLA, Tokyo Electron, Zeiss (optics), specialty chemicals, photoresists.

Business example: TSMC is the world’s leading pure-play foundry — it manufactures chips for many fabless firms and invests aggressively in node and capacity expansion.

Market figures: the semiconductor market is large and projected to continue growing with strong CAGR into the late 2020s. 
Mordor Intelligence
+1

6. Recent shocks & supply-chain lessons (COVID shortage → policy responses)

The COVID-era shortage (2020–2022) exposed fragility: demand surges (remote work, gaming, automotive electronics) and capacity limits led to global shortages. Automotive OEMs saw production cuts, long lead times, and feature deletions when specific parts were unavailable. The shortage triggered national policy responses — subsidies, incentives, and on-shoring initiatives in the U.S., Europe and Asia to reduce concentration risk and increase strategic capacity. 
ScienceDirect
+1

Example: Several carmakers temporarily removed infotainment or ADAS features because the required microcontrollers weren’t available, delaying vehicle production.

7. Geopolitics and strategic technology competition

Semiconductors have become central to national security and economic competitiveness. Key dynamics:

Concentration risk: advanced logic capacity is concentrated in Taiwan (TSMC) and South Korea (Samsung).

Export controls: the U.S. has restricted some equipment and advanced chips to certain countries (notably China), reshaping supply-chain flows.

Industrial policy: governments have launched CHIPS acts or equivalent funding to build domestic capacity.

Technology diffusion & reverse engineering: there are reports of states attempting to replicate advanced equipment (e.g., prototypes of EUV-like systems) — a process that is technologically fraught and may take many years to reach parity. 
Tom's Hardware
+1

These tensions are driving diversification of supply chains and investment in regional fabs.

8. Economics & the cost of scaling (why “more than miniaturization”)

Shrinking to a new process node historically improved performance and density, but the cost per transistor improvements have slowed: fabs and equipment costs rise steeply, and R&D investment is enormous. As a result, companies are balancing node scaling with architecture innovation, specialized accelerators, and packaging strategies that deliver system performance without always chasing the next node.

Example: instead of always using the smallest node for every component, many designers put logic on an advanced node but move analog or I/O to more mature (cheaper) nodes, balancing performance and cost.

9. Advanced packaging and the chiplet movement (examples & impact)

As monolithic scaling becomes harder and more expensive, heterogeneous integration (chiplets, 2.5D/3D stacking, interposers, fan-out packages) is rising. Chiplets are small dies (IP blocks) integrated into a package to act like a single SoC but with independent process choices for each chiplet. Advanced packaging improves yield (smaller dies), reduces time-to-market, and lets designers mix logic, memory, and analog IP from different process nodes.

Market and trend data indicate a sharp rise in advanced packaging adoption and revenue — packaging is becoming a strategic battleground. 
Yole Group
+1

Example: High-end GPUs and some data-center accelerators adopt multi-chip tiles with HBM (stacked memory) and compute chiplets on an interposer to cheaply scale compute and memory bandwidth.

10. Application drivers: why demand keeps rising (AI, automotive, mobile, edge)

AI & data centers: training and inference workloads require massive compute and memory bandwidth, fueling demand for specialized accelerators (GPUs, TPUs, AI ASICs) and HBM memory.

Automotive: EV power electronics, battery management, ADAS/autonomy and infotainment add semiconductor content per vehicle.

Mobile & edge: 5G, AR/VR and always-on sensing push integrated radios, baseband processors and low-power AI inference at the edge.

Industrial & IoT: sensing, control and connectivity across manufacturing and infrastructure add diverse chip demand.

Example: A single generative AI training cluster can consume orders of magnitude more silicon area (and power) than a typical cloud server workload, increasing demand for top-end logic and memory chips.

11. Energy, materials and sustainability concerns

Chip fabs are energy- and water-intensive: ultra-pure water, specialty gases, and temperature-controlled cleanrooms are necessary. As fabs expand, sustainability becomes critical — fabs source renewables, recycle water, and optimize processes to reduce per-wafer energy. Lifecycle emissions also depend on packaging and system integration.

Example actions: leading foundries report investments in renewable sourcing and water recycling to reduce environmental footprint.

12. Near-term technology frontiers (what to watch)

High-NA EUV & next-gen lithography — further resolution gains but high cost & complexity. 
Tom's Hardware
+1

Heterogeneous integration & chiplets — system scaling using modular dies and advanced interconnect. 
PatentPC
+1

Packaging & memory stacking (HBM) — to meet bandwidth needs of AI workloads.

Materials & transistor architectures — GAA (gate all around), nanosheets, and research into beyond-CMOS devices for special uses.

Regional capacity expansion — new fabs in the U.S., Europe, China and elsewhere, shifted by policy & investment. 
Reuters

13. Industry risks and constraints (concise)

Capital intensity & concentration: enormous capex and concentration of advanced nodes in a few geographies.

Supply-chain fragility: single points (equipment vendors, specialized materials).

Geopolitical tensions: export controls and supply re-routing.

Physical limits: at extreme scaling, quantum & variability effects require new device concepts or integration strategies.

Ecosystem skills: fab-level expertise and supply-chain partners are hard to build quickly.

14. Strategic responses (how companies & governments are reacting)

Diversification: building fabs in multiple regions, on-shoring where strategic.

Public funding & incentives: CHIPS acts and subsidies to accelerate domestic capacity.

Platform specialization: more domain-specific accelerators rather than general scaling-first strategies.

Partnerships: co-investments between governments and fabs, foundry-customer agreements, and cross-industry alliances.

Example: several governments now offer incentives to attract leading-edge fabs, and foundries announce multi-billion dollar plants with public support.

15. Conclusion — synthesis and outlook (practical takeaways)

Semiconductors remain central to technology and geopolitical strategy. The market is large and expected to grow as AI, automotive electrification, and pervasive sensing expand chip content per system. 
Mordor Intelligence
+1

Technology progress continues but is diversifying: traditional node scaling (Moore’s Law) coexists with heterogenous integration (chiplets), advanced packaging, and better system architecture to get more performance per dollar and per watt. 
BCG Global

Supply-chain resilience, energy & water sustainability, and regional capacity are priority areas for industry planners and policymakers. The balance of power in equipment and materials (e.g., ASML for EUV) remains strategically important. 
ASML
+1

If you need, I can:

expand any section into a standalone deep-dive (e.g., EUV lithography, chiplet architectures, or the semiconductor supply chain) with diagrams and concrete case studies; or

produce a slide deck summarizing this report for presentations (3–10 slides), or a one-page executive brief.

Short reading list / sources (selected)

PwC — State of the semiconductor industry (industry trends & forecasts). 
PwC

Mordor Intelligence — Semiconductor market forecasts. 
Mordor Intelligence

ASML — EUV product pages (technical overview). 
ASML

Reuters / Tom’s Hardware — recent reporting on equipment, geopolitics and developments. 
Reuters
+1

BCG & Yole Group — advanced packaging insights and market forecasts. 
BCG Global



'''
# Invoke
result = chain.invoke({'text':text})
print(result)

# Visualise Chain
chain.get_graph().print_ascii() 



#  Chain Visualisation
'''
          +---------------------------+
          | Parallel<notes,quiz>Input |
          +---------------------------+
                ***             ***
              **                   **
+----------------+              +----------------+
| PromptTemplate |              | PromptTemplate |
+----------------+              +----------------+
          *                             *
    +----------+                  +----------+
    | ChatGroq |                  | ChatGroq |
    +----------+                  +----------+
          *                             *
+-----------------+            +-----------------+
| StrOutputParser |            | StrOutputParser |
+-----------------+            +-----------------+
                   **         **
                     **     ** 
          +----------------------------+
          | Parallel<notes,quiz>Output |
          +----------------------------+
                         *
                +----------------+
                | PromptTemplate |
                +----------------+
                         *
                   +----------+
                   | ChatGroq |
                   +----------+
                         *
                +-----------------+
                | StrOutputParser |
                +-----------------+
                         *
            +-----------------------+
            | StrOutputParserOutput |
            +-----------------------+


'''


# # OUTPUT
'''
**Semiconductor Industry Report Notes**

**Executive Summary**

* Semiconductors are the foundation of modern electronics, controlling current flow and enabling computation, sensing, and communication.
* The industry has evolved over 80 years, driven by innovation in device physics, materials, lithography, and packaging.
* The market is large and growing, driven by megatrends like AI, cloud, mobile computing, automotive electrification, and IoT.

**Section 1: What is a Semiconductor?**

* A semiconductor is a material with electrical conductivity between conductors and insulators, with silicon being the dominant material.
* Semiconductors are engineered into devices like diodes and transistors through doping, dielectric layers, and patterning metal interconnects.
* MOSFET (metal-oxide-semiconductor field-effect transistor) is the most important device family for modern logic and memory.    

**Section 2: Short History and Moore's Law**

* Key milestones in the industry's history include the point-contact transistor (1947), silicon integrated circuit (late 1950s), and planar MOS processes (1960s).
* Moore's Law observes that transistor density roughly doubles every 18-24 months, driving the industry's growth.
* Process node refers to the minimum feature size printed on chip layers, but is now a marketing term tied to transistor performance and density.

**Section 3: How Chips are Made**

* The manufacturing chain involves design, mask creation, wafer fab, BEOL/metallization, testing and packaging, and distribution and system integration.
* Wafer fab is the most technology-intensive and capital-heavy part of the process, with modern fabs costing billions of dollars.
* Lithography is the critical enabler of feature size reduction, with EUV lithography being a key technology for advanced nodes. 

**Section 4: Lithography and EUV**

* Lithography prints circuit patterns on photoresist using light, with EUV lithography being a key technology for advanced nodes.
* ASML is the dominant supplier of EUV scanners and therefore a pivotal supplier in the ecosystem.
* EUV has enabled simpler patterning for advanced nodes, but is complex and expensive to deploy.

**Section 5: Industry Structure and Leading Players**

* The industry has specialized players across the value chain, including IDMs, foundries, fabless designers, EDA & IP providers, equipment makers, and materials suppliers.
* TSMC is the world's leading pure-play foundry, manufacturing chips for many fabless firms and investing aggressively in node and capacity expansion.

**Section 6: Recent Shocks and Supply-Chain Lessons**

* The COVID-era shortage exposed fragility in the industry, with demand surges and capacity limits leading to global shortages.  
* Automotive OEMs saw production cuts, long lead times, and feature deletions when specific parts were unavailable.
* National policy responses included subsidies, incentives, and on-shoring initiatives to reduce concentration risk and increase strategic capacity.

**Section 7: Geopolitics and Strategic Technology Competition**

* Semiconductors have become central to national security and economic competitiveness, with concentration risk, export controls, and industrial policy being key dynamics.
* Governments have launched CHIPS acts or equivalent funding to build domestic capacity.
* Technology diffusion and reverse engineering are also key concerns.

**Section 8: Economics and the Cost of Scaling**

* Shrinking to a new process node historically improved performance and density, but the cost per transistor improvements have slowed.
* Companies are balancing node scaling with architecture innovation, specialized accelerators, and packaging strategies to deliver system performance without always chasing the next node.

**Section 9: Advanced Packaging and the Chiplet Movement**

* As monolithic scaling becomes harder and more expensive, heterogeneous integration (chiplets, 2.5D/3D stacking, interposers, fan-out packages) is rising.
* Chiplets are small dies (IP blocks) integrated into a package to act like a single SoC but with independent process choices for each chiplet.
* Advanced packaging improves yield, reduces time-to-market, and lets designers mix logic, memory, and analog IP from different process nodes.

**Section 10: Application Drivers**

* AI & data centers: training and inference workloads require massive compute and memory bandwidth, fueling demand for specialized accelerators (GPUs, TPUs, AI ASICs) and HBM memory.
* Automotive: EV power electronics, battery management, ADAS/autonomy, and infotainment add semiconductor content per vehicle.   
* Mobile & edge: 5G, AR/VR, and always-on sensing push integrated radios, baseband processors, and low-power AI inference at the edge.

**Section 11: Energy, Materials, and Sustainability Concerns**

* Chip fabs are energy- and water-intensive, requiring ultra-pure water, specialty gases, and temperature-controlled cleanrooms. 
* Sustainability becomes critical as fabs expand, with leading foundries reporting investments in renewable sourcing and water recycling to reduce environmental footprint.

**Section 12: Near-Term Technology Frontiers**

* High-NA EUV & next-gen lithography: further resolution gains but high cost & complexity.
* Heterogeneous integration & chiplets: system scaling using modular dies and advanced interconnect.
* Packaging & memory stacking (HBM): to meet bandwidth needs of AI workloads.
* Materials & transistor architectures: GAA, nanosheets, and research into beyond-CMOS devices for special uses.

**Section 13: Industry Risks and Constraints**

* Capital intensity & concentration: enormous capex and concentration of advanced nodes in a few geographies.
* Supply-chain fragility: single points (equipment vendors, specialized materials).
* Geopolitical tensions: export controls and supply re-routing.
* Physical limits: at extreme scaling, quantum & variability effects require new device concepts or integration strategies.      
* Ecosystem skills: fab-level expertise and supply-chain partners are hard to build quickly.

**Section 14: Strategic Responses**

* Diversification: building fabs in multiple regions, on-shoring where strategic.
* Public funding & incentives: CHIPS acts and subsidies to accelerate domestic capacity.
* Platform specialization: more domain-specific accelerators rather than general scaling-first strategies.
* Partnerships: co-investments between governments and fabs, foundry-customer agreements, and cross-industry alliances.

**Conclusion**

* Semiconductors remain central to technology and geopolitical strategy.
* The market is large and expected to grow as AI, automotive electrification, and pervasive sensing expand chip content per system.
* Technology progress continues but is diversifying, with traditional node scaling coexisting with heterogeneous integration, advanced packaging, and better system architecture.
* Supply-chain resilience, energy & water sustainability, and regional capacity are priority areas for industry planners and policymakers.
* The balance of power in equipment and materials remains strategically important.

**Quiz**

1. **What is a semiconductor?**

A semiconductor is a material whose electrical conductivity lies between that of a conductor and an insulator, with silicon being the dominant semiconductor material. It is engineered into devices such as diodes and transistors by doping, building dielectric layers, and patterning metal interconnects.

2. **What is the meaning of “node” in the semiconductor industry?**

Historically, a "process node" referred to the minimum feature printed on chip layers, but today it's more a marketing/capability label tied to transistor performance, density, and power than a literal single dimension.

3. **What is the main challenge facing the semiconductor industry in terms of manufacturing?**

The main challenge facing the semiconductor industry is the cost and complexity of sub-nm scaling, as well as the rising cost of fabs and equipment, which makes it difficult to improve performance and density while reducing costs.

4. **What is the significance of EUV lithography in the semiconductor industry?**

EUV lithography (Extreme Ultraviolet lithography) enables simpler patterning for advanced nodes (7/5/3 nm) and is critical for the production of leading-edge chips. ASML is the dominant supplier of EUV scanners and is a pivotal supplier in the ecosystem.    

5. **What is driving the growth of the semiconductor market?**

The growth of the semiconductor market is driven by megatrends such as artificial intelligence (AI), cloud/data centers, mobile c

EUV lithography (Extreme Ultraviolet lithography) enables simpler patterning for advanced nodes (7/5/3 nm) and is critical for the production of leading-edge chips. ASML is the dominant supplier of EUV scanners and is a pivotal supplier in the ecosystem.    

5. **What is driving the growth of the semiconductor market?**

The growth of the semiconductor market is driven by megatrends such as artificial intelligence (AI), cloud/data centers, mobile computing, automotive electrification/autonomy, and ubiquitous sensing (IoT), which are all increasing chip content per system.


'''