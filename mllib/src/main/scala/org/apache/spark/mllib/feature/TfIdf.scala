package statistic


import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext


object TfIdf {
  /**
   compute doc tf-idf value,
   **/
  def computeTFIDFOneByOne(sc:SparkContext,data:RDD[(String, Array[String])]):RDD[(String, Array[(String, Double)])]={

         import scala.collection.JavaConverters._
         import breeze.linalg._

         val dataBroad = sc.broadcast(data.collect())

         val result = data.map{ from =>

              val docWordCount = from._2.groupBy {w => w}.map{ e =>(e._1,e._2.length)}
              val total = dataBroad.value.size.toDouble

              val docId = from._1
              val words = from._2.toSet
              val res = dataBroad.value.filter(_._1 != docId)
              val returnData = words.map { word =>
                    var countIDF = 1.0
                    res.foreach{ otherDoc=>
                      if (otherDoc._2.toSet.contains(word)){
                              countIDF = countIDF + 1.0
                      }
                    }
                    val tdidf = docWordCount.get(word).get*Math.log((total+1.0)/(countIDF+1.0))
                    (word,tdidf)
              }
             (docId,returnData.toArray)
          }
      result
  }

  /**
   compute doc TF-IDF vector,
   **/
  def computeTFIDFVector(sc:SparkContext,tfidfs:RDD[(String,Array[(String,Double)])]):RDD[(String, Vector)]={

      val allWords = sc.broadcast(computeWordsDictionary(tfidfs).collect())

      val TFIDF_MODEL = tfidfs.map{ value =>
             val docId = value._1
             val idvalueMap = value._2.map{ x=>
                val i = allWords.value.indexOf(x._1)
                (i,x._2)
             }.sortBy(_._1)
             val vec = Vectors.sparse(allWords.value.size, idvalueMap)
             (docId,vec)
          }
      TFIDF_MODEL
  }

  /**
    get all text word,and distinct
    **/
  def computeWordsDictionary(data:RDD[(String,Array[(String,Double)])]):RDD[String]={
       data.flatMap{doc =>doc._2.map(_._1)}.distinct()
  }
}
