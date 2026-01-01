from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch


from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed Report on Topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summaries the {topic}',
    input_variables=['topic']
)

#  Chain 1 : Report Generation
report_chain = prompt1 | llm | parser

#  Sumarry chain 
summary_chain =  prompt2 | llm | parser

def word_count_checker(text: str) -> bool:
    word_count = len(text.split())
    print("\n🔹 Original Word Count:", word_count)
    return word_count > 300

# ---------------- Print original ----------------
print_original = RunnableLambda(
    lambda text: (print("\n📄 ORIGINAL TEXT (Long Report):\n", text),
        text
    )[1]
)
#  Branch Chain
branch_chain = RunnableBranch(
    (word_count_checker, print_original | summary_chain),  # Condition Function
    RunnablePassthrough(),  # If less than 500 words, return as it is
)


#  Final Chain : Report Generation + Branching
final_chain = report_chain | branch_chain

result = final_chain.invoke({'topic':'Russia vs Ukraine War'})
print("\n\n\n\nFinal Output:\n", result)

print("\n\n\n Length of final Output:\n", len(result.split()))

#  Visualise Chain
final_chain.get_graph().print_ascii()



#  OUTPUT 
'''
🔹 Original Word Count: 1033

📄 ORIGINAL TEXT (Long Report):
 **Report: Russia-Ukraine War**

**Introduction**

The Russia-Ukraine War is an ongoing conflict between Russia and Ukraine that began on February 24, 2022, when Russia launched a large-scale invasion of Ukraine. The conflict has resulted in significant human suffering, displacement of people, and economic destruction. This report provides a detailed analysis of the background, causes, and current situation of the conflict.

**Background**

The conflict between Russia and Ukraine has its roots in the country's history. Ukraine was part of the Soviet Union until its dissolution in 1991, and Russia has maintained close economic and cultural ties with Ukraine ever since. However, Ukraine's pro-European Union (EU) and pro-NATO policies have been a source of tension between the two countries.

**Causes of the Conflict**

The conflict in Ukraine began in 2014, when pro-Russian President Viktor Yanukovych was ousted following months of protests in Kiev. The new government, led by pro-EU President Petro Poroshenko, implemented a series of reforms aimed at strengthening ties with the EU and NATO. Russia responded by annexing Crimea, a peninsula in southern Ukraine, in March 2014. The annexation was widely condemned by the international community, and the conflict in eastern Ukraine began soon after.

**Escalation of the Conflict**

The conflict in eastern Ukraine escalated in 2014, with pro-Russian separatists in Donetsk and Luhansk declaring independence from Ukraine. The Ukrainian government responded with military force, leading to a protracted conflict that has resulted in thousands of deaths and injuries. Russia has been accused of providing military support to the separatists, including troops, arms, and equipment.

**Russia's Invasion of Ukraine**

On February 24, 2022, Russia launched a large-scale invasion of Ukraine, with troops and tanks pouring into the country from multiple directions. The invasion was widely condemned by the international community, and Ukraine's government called for international support. The conflict has resulted in significant human suffering, with thousands of civilians killed or injured, and millions displaced.

**Current Situation**

The conflict in Ukraine is ongoing, with both sides suffering significant losses. Ukraine's government has launched a counterattack against Russian forces, but the situation remains fluid and uncertain. The international community has imposed severe economic sanctions on Russia, and the country's economy is in crisis.

**Humanitarian Situation**

The humanitarian situation in Ukraine is dire, with millions of people displaced and in need of assistance. The conflict has resulted in significant damage to infrastructure, including homes, schools, and hospitals. Food and water shortages are widespread, and the risk of disease and starvation is high.

**International Response**

The international community has responded to the conflict in Ukraine with widespread condemnation and economic sanctions. The United States, European Union, and other countries have imposed severe economic sanctions on Russia, including freezing assets and restricting trade. The United Nations has also played a key role in mediating the conflict, with Secretary-General António Guterres calling for a ceasefire and diplomatic talks.

**Key Players**

* **Russia**: Russia's government has been accused of launching the invasion of Ukraine, and the country's military has been responsible for significant human suffering.
* **Ukraine**: Ukraine's government has called for international support and has launched a counterattack against Russian forces.
* **United States**: The United States has imposed severe economic sanctions on Russia and has provided military aid to Ukraine.
* **European Union**: The European Union has imposed economic sanctions on Russia and has called for a ceasefire and diplomatic talks.
* **United Nations**: The United Nations has played a key role in mediating the conflict and has called for a ceasefire and diplomatic talks.

**Key Events**

* **February 24, 2022**: Russia launches a large-scale invasion of Ukraine.
* **March 2022**: Ukraine launches a counterattack against Russian forces.
* **April 2022**: The international community imposes severe economic sanctions on Russia.
* **May 2022**: The United Nations calls for a ceasefire and diplomatic talks.

**Conclusion**

The Russia-Ukraine War is a complex and ongoing conflict that has resulted in significant human suffering, displacement of people, and economic destruction. The conflict has its roots in the country's history and has been exacerbated by Russia's annexation of Crimea and its military support for separatists in eastern Ukraine. The international community has responded to the conflict with widespread condemnation and economic sanctions, but the situation remains fluid and uncertain.

**Recommendations**

* **Diplomatic talks**: The international community should continue to call for diplomatic talks between Russia and Ukraine.
* **Economic sanctions**: The international community should maintain and strengthen economic sanctions on Russia to pressure the country to withdraw its troops.
* **Humanitarian aid**: The international community should provide significant humanitarian aid to Ukraine to support the country's displaced people and damaged infrastructure.
* **Long-term solution**: The international community should work towards a long-term solution to the conflict, including a negotiated settlement and a commitment to Ukraine's sovereignty and territorial integrity.

**Timeline**

* **February 24, 2022**: Russia launches a large-scale invasion of Ukraine.
* **March 2022**: Ukraine launches a counterattack against Russian forces.
* **April 2022**: The international community imposes severe economic sanctions on Russia.
* **May 2022**: The United Nations calls for a ceasefire and diplomatic talks.
* **June 2022**: The conflict continues, with both sides suffering significant losses.
* **July 2022**: The international community continues to support Ukraine with economic aid and military assistance.
* **August 2022**: The conflict continues, with both sides suffering significant losses.
* **September 2022**: The United Nations calls for a renewed ceasefire and diplomatic talks.
* **October 2022**: The conflict continues, with both sides suffering significant losses.
* **November 2022**: The international community continues to support Ukraine with economic aid and military assistance.
* **December 2022**: The conflict continues, with both sides suffering significant losses.

**Bibliography**

* **The New York Times**: "Russia Invades Ukraine, Sparking Global Condemnation"
* **The Washington Post**: "Ukraine Launches Counterattack Against Russian Forces"
* **BBC News**: "Russia's Invasion of Ukraine: Timeline"
* **Al Jazeera**: "Ukraine-Russia Conflict: What You Need to Know"
* **The Guardian**: "Ukraine Conflict: Millions Displaced, Humanitarian Crisis Deepens"

**Appendix**

* **Map of Ukraine**: A map showing Ukraine's borders and the conflict zones.
* **Timeline of Events**: A detailed timeline of the conflict, including key events and dates.
* **List of Key Players**: A list of key players involved in the conflict, including Russia, Ukraine, the United States, and the European Union.




Final Output:
 **Summary of the Report: Russia-Ukraine War**

The report provides a detailed analysis of the ongoing conflict between Russia and Ukraine, which began on February 24, 2022, when Russia launched a large-scale invasion of Ukraine. The conflict has resulted in significant human suffering, displacement of people, and economic destruction.

**Background and Causes of the Conflict**

The conflict has its roots in Ukraine's history, dating back to its independence from the Soviet Union in 1991. Ukraine's pro-European Union (EU) and pro-NATO policies have been a source of tension between Russia and Ukraine. The conflict escalated in 2014, when pro-Russian President Viktor Yanukovych was ousted, and Russia annexed Crimea. This led to a protracted conflict in eastern Ukraine, with pro-Russian separatists declaring independence from Ukraine.

**Escalation of the Conflict and Russia's Invasion of Ukraine**

The conflict escalated further in 2022, when Russia launched a large-scale invasion of Ukraine, with troops and tanks pouring into the country from multiple directions. The international community condemned the invasion, and Ukraine's government called for international support. The conflict has resulted in significant human suffering, with thousands of civilians killed or injured, and millions displaced.

**Current Situation and Humanitarian Situation**

The conflict is ongoing, with both sides suffering significant losses. Ukraine's government has launched a counterattack against Russian forces, but the situation remains fluid and uncertain. The humanitarian situation in Ukraine is dire, with millions of people displaced and in need of assistance. The conflict has resulted in significant damage to infrastructure, including homes, schools, and hospitals.

**International Response and Key Players**

The international community has responded to the conflict with widespread condemnation and economic sanctions. The United States, European Union, and other countries have imposed severe economic sanctions on Russia, including freezing assets and restricting trade. The United Nations has also played a key role in mediating the conflict, with Secretary-General António Guterres calling for a ceasefire and diplomatic talks. Key players involved in the conflict include Russia, Ukraine, the United States, the European Union, and the United Nations.

**Recommendations and Timeline**

The report recommends diplomatic talks between Russia and Ukraine, maintaining and strengthening economic sanctions on Russia, providing humanitarian aid to Ukraine, and working towards a long-term solution to the conflict. A detailed timeline of the conflict is provided, including key events and dates.

**Bibliography and Appendix**

The report cites various news sources, including The New York Times, The Washington Post, BBC News, Al Jazeera, and The Guardian. An appendix provides a map of Ukraine, a timeline of events, and a list of key players involved in the conflict.



 Length of final Output:
 423

'''





#  Chain Visualization
'''

  +-------------+    
  | PromptInput |
  +-------------+
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
    +--------+
    | Branch |
    +--------+
          *
  +--------------+
  | BranchOutput |
  +--------------+

'''