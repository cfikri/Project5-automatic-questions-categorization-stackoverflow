A fan of **Stack Overflow**, which has often saved the day for you, you decide to give back to the community. You reach out to the person in charge of Stack Overflow to find out how you can contribute.

He replied via email:

"Hello,

What a pleasure to hear that you appreciate the site. That’s perfect timing because I’ve been looking to improve tag management, as users sometimes define them incorrectly when creating their questions. This isn’t an issue for experienced users, but for newcomers, it would be helpful to suggest a few relevant tags based on their question.

**Could you develop an ***automatic tag suggestion system*** for this purpose? A machine learning algorithm could automatically assign multiple relevant tags to a question. You would only need to create the algorithm of course, our developers will handle the interface.**

Stack Overflow provides a **data export tool** using SQL queries, called ***"StackExchange Data Explorer"***, which allows us to collect data from the platform. You can use this tool to extract up to 50,000 questions, which is the limit per query.

Additionally, be mindful of **question quality and tag completeness**, as they can be highly inconsistent, which might bias your model’s results. To mitigate this risk and obtain the most relevant results possible, consider using queries with constraints to filter questions based on factors such as:

The most viewed questions,
questions marked as favorites or deemed relevant by users,
questions that have received an answer,
questions with at least five tags.

There is also an API that allows us to retrieve information about Stack Overflow questions. I’d like to evaluate the relevance of this API and the quality of its documentation.

So, as a test, **could you ***run a query in a notebook*** to retrieve ***50 questions*** from a defined time period that contain the "python" tag, have a score > 50 (votes), extract the main details of each question (date, title, tags, score) into a DataFrame and display them ?**

You can use either the **StackAPI Python wrapper** or directly the **Stack Exchange API**. Your feedback on its usability would be valuable in determining whether any improvements are needed.

Also, please make sure to **comply with GDPR principles** for personal data management. In particular, only collect the data necessary for the project and avoid storing any information that could identify the authors of the questions."

**Key skills involved :**

- Querying a database with SQL
- NLP techniques
- Topic modeling
- Implementation and comparison of algorithms
- Hyperparameter optimization
- Model management, deployment, and monitoring using MLflow and GitHub
- Creating and deploying a web API with Streamlit and Render
