import { useEffect, useRef } from "react";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
} from "chart.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale
);

const PRROCChart = ({ prData, rocData }) => {
  const prChartRef = useRef(null);
  const rocChartRef = useRef(null);
  const prChartInstance = useRef(null);
  const rocChartInstance = useRef(null);

  useEffect(() => {
    // Clean up the previous chart instance before creating a new one
    if (prChartInstance.current) {
      prChartInstance.current.destroy();
    }

    if (prChartRef.current && prData) {
      const prCtx = prChartRef.current.getContext("2d");
      prChartInstance.current = new Chart(prCtx, {
        type: "line",
        data: {
          datasets: [
            {
              label: "Precision-Recall",
              data: prData.map((d) => ({ x: d.recall, y: d.precision })),
              borderColor: "green",
              fill: false,
            },
          ],
        },
        options: {
          responsive: true, // Ensure responsiveness
          maintainAspectRatio: false,
          scales: {
            x: {
              type: "linear",
              min: 0,
              max: 1,
              title: {
                display: true,
                text: "Recall",
              },
              ticks: {
                padding: 10, // Add padding to x-axis labels
              },
            },
            y: {
              type: "linear",
              min: 0,
              max: 1,
              title: {
                display: true,
                text: "Precision",
              },
              ticks: {
                padding: 10, // Add padding to x-axis labels
              },
            },
          },
          elements: {
            line: {
              tension: 0.4, // Add this to smooth the lines (0-1 range)
            },
            point: {
              radius: function (context) {
                var index = context.dataIndex;
                // Show points only at every 10th interval
                return index % 10 === 0 ? 5 : 0;
              },
            },
          },
        },
      });
    }

    // Clean up the previous ROC chart instance before creating a new one
    if (rocChartInstance.current) {
      rocChartInstance.current.destroy();
    }

    if (rocChartRef.current && rocData) {
      const rocCtx = rocChartRef.current.getContext("2d");
      rocChartInstance.current = new Chart(rocCtx, {
        type: "line",
        data: {
          datasets: [
            {
              label: "ROC",
              data: rocData.map((d) => ({ x: d.fpr, y: d.tpr })),
              borderColor: "blue",
              fill: false,
            },
          ],
        },
        options: {
          responsive: true, // Ensure responsiveness
          maintainAspectRatio: false,
          scales: {
            x: {
              type: "linear",
              min: 0,
              max: 1,
              title: {
                display: true,
                text: "False Positive Rate",
              },
              ticks: {
                padding: 10, // Add padding to x-axis labels
              },
            },
            y: {
              type: "linear",
              min: 0,
              max: 1,
              title: {
                display: true,
                text: "True Positive Rate",
              },
              ticks: {
                padding: 10, // Add padding to x-axis labels
              },
            },
          },
          elements: {
            line: {
              tension: 0.4, // Add this to smooth the lines (0-1 range)
            },
            point: {
              radius: function (context) {
                var index = context.dataIndex;
                // Show points only at every 10th interval
                return index % 10 === 0 ? 5 : 0;
              },
            },
          },
        },
      });
    }

    // Cleanup function to destroy charts on component unmount
    return () => {
      if (prChartInstance.current) prChartInstance.current.destroy();
      if (rocChartInstance.current) rocChartInstance.current.destroy();
    };
  }, [prData, rocData]);

  return (
    <div>
      <h2 className="text-xl font-bold mt-4 mb-2">Precision-Recall Curve</h2>
      <div class="chart-container">
        <canvas ref={prChartRef} width="400" height="200"></canvas>
      </div>

      <h2 className="text-xl font-bold mt-4 mb-2">ROC Curve</h2>
      <div class="chart-container">
        <canvas ref={rocChartRef} width="400" height="200"></canvas>
      </div>
    </div>
  );
};

export default PRROCChart;
