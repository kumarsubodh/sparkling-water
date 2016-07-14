package hex;

import hex.AUC2;
import hex.ModelMetricsBinomial;
import hex.ModelMetricsRegression;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.spark.models.svm.SVMModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;
import water.fvec.Frame;

public class SVMDriverUtil {

    public static void addModelMetrics(SVMModel model, RDD<LabeledPoint> training,
                                       final org.apache.spark.mllib.classification.SVMModel trainedModel,
                                       Frame f,
                                       String[] responseDomains) {

        // Compute Spark evaluations
        JavaRDD<Tuple2<Double, Double>> predictionAndLabels = training.toJavaRDD().map(
                new Function<LabeledPoint, Tuple2<Double, Double>>() {
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        Double prediction = trainedModel.predict(p.features());
                        return new Tuple2<>(prediction, p.label());
                    }
                }
        );

        ModelMetricsBinomial.MetricBuilderBinomial builder = new ModelMetricsBinomial.MetricBuilderBinomial(responseDomains);
        for (Tuple2<Double, Double> predAct : predictionAndLabels.collect()) {
            Double pred = predAct._1;
            builder.perRow(new double[] {pred, pred == 1 ? 0 : 1, pred == 1 ? 1 : 0}, new float[] {predAct._2.floatValue()}, model);
        }
        double mse = builder._sumsqe / builder._nclasses;

        // Set the metrics
        switch (model._output.getModelCategory()) {
            case Binomial:
                model._output._training_metrics =
                new ModelMetricsBinomial(
                        model, f, mse, responseDomains,
                        builder.weightedSigma(), new AUC2(builder._auc), builder._logloss, null
                );
                break;
            default:
                model._output._training_metrics =
                        new ModelMetricsRegression(
                                model, f, mse, builder.weightedSigma(), 0
                        );
                break;
        }
    }

}
