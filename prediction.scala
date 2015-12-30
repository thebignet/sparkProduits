import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

// Load and parse the data file
val data = sc.textFile("produits.csv")
//val data = sc.textFile("produitsADeviner.csv")
val parsedData = data.map { line =>
  val parts = line.split(',').tail.map(_.toDouble)
  Vectors.dense(parts)
}

val model = DecisionTreeModel.load(sc, "myModel")

// Evaluate model on training examples and compute training error
val predictions = parsedData.map { vector =>
  model.predict(vector)
}

predictions.foreach(println)

System.exit(0)