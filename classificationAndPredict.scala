import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

// Load and parse the data file
val data = sc.textFile("produits.csv")
val parsedData = data.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

// Run training algorithm to build the model
val maxDepth = 5
val numClasses = 3
val model = DecisionTree.train(parsedData, Classification, Gini, maxDepth, numClasses)

// Evaluate model on training examples and compute training error
val labelAndPreds = parsedData.map { point =>
  val prediction = model.predict(point.features)
  println("label : "+point.label+" prediction : "+prediction)
  (point.label, prediction)
}
val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
println("Training Error = " + trainErr)

//val data = sc.textFile("produits.csv")
val data = sc.textFile("produitsADeviner.csv")
val predictions = data.map { line =>
  model.predict(Vectors.dense(line.split(',').tail.map(_.toDouble)))
}

predictions.collect().foreach(println)

System.exit(0)