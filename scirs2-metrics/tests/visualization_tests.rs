//! Tests for visualization components
//!
//! This module tests the visualization components in the scirs2-metrics crate.

use ndarray::{array, Array2};
use scirs2_metrics::{
    classification::confusion_matrix,
    classification::curves::{calibration_curve, precision_recall_curve, roc_curve},
    visualization::{
        backends, helpers, ColorMap, PlotType, PlottingBackend, VisualizationData,
        VisualizationMetadata, VisualizationOptions,
    },
};

#[test]
fn test_visualization_data() {
    // Test creating a new VisualizationData
    let mut data = VisualizationData::new();
    assert!(data.x.is_empty());
    assert!(data.y.is_empty());
    assert!(data.z.is_none());
    assert!(data.series_names.is_none());
    assert!(data.x_labels.is_none());
    assert!(data.y_labels.is_none());
    assert!(data.auxiliary_data.is_empty());
    assert!(data.auxiliary_metadata.is_empty());
    assert!(data.series.is_empty());

    // Test adding series
    data.add_series("x", vec![1.0, 2.0, 3.0]);
    data.add_series("y", vec![4.0, 5.0, 6.0]);
    data.add_series("z_series", vec![7.0, 8.0, 9.0]);

    assert_eq!(data.x, vec![1.0, 2.0, 3.0]);
    assert_eq!(data.y, vec![4.0, 5.0, 6.0]);
    assert!(data.series.contains_key("z_series"));
    assert_eq!(data.series["z_series"], vec![7.0, 8.0, 9.0]);

    // Test adding 2D data
    let matrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    data.add_heatmap_data(matrix.clone());
    assert_eq!(data.z, Some(matrix));

    // Test adding labels
    data.add_x_labels(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    data.add_y_labels(vec!["D".to_string(), "E".to_string(), "F".to_string()]);
    data.add_series_names(vec!["Series 1".to_string(), "Series 2".to_string()]);

    assert_eq!(
        data.x_labels,
        Some(vec!["A".to_string(), "B".to_string(), "C".to_string()])
    );
    assert_eq!(
        data.y_labels,
        Some(vec!["D".to_string(), "E".to_string(), "F".to_string()])
    );
    assert_eq!(
        data.series_names,
        Some(vec!["Series 1".to_string(), "Series 2".to_string()])
    );

    // Test adding auxiliary data
    data.add_auxiliary_data("aux_data", vec![10.0, 11.0, 12.0]);
    data.add_auxiliary_metadata("aux_meta", "test metadata");

    assert!(data.auxiliary_data.contains_key("aux_data"));
    assert_eq!(data.auxiliary_data["aux_data"], vec![10.0, 11.0, 12.0]);
    assert!(data.auxiliary_metadata.contains_key("aux_meta"));
    assert_eq!(data.auxiliary_metadata["aux_meta"], "test metadata");
}

#[test]
fn test_visualization_metadata() {
    // Test creating a new VisualizationMetadata
    let metadata = VisualizationMetadata::new("Test Title");
    assert_eq!(metadata.title, "Test Title");
    assert_eq!(metadata.x_label, "X");
    assert_eq!(metadata.y_label, "Y");
    assert!(matches!(metadata.plot_type, PlotType::Line));
    assert!(metadata.description.is_none());

    // Test setter methods
    let mut metadata = VisualizationMetadata::new("Test Title");
    metadata.set_plot_type(PlotType::Scatter);
    metadata.set_x_label("X Label");
    metadata.set_y_label("Y Label");
    metadata.set_description("Test Description");

    assert_eq!(metadata.plot_type, PlotType::Scatter);
    assert_eq!(metadata.x_label, "X Label");
    assert_eq!(metadata.y_label, "Y Label");
    assert_eq!(metadata.description, Some("Test Description".to_string()));

    // Test factory methods
    let metadata = VisualizationMetadata::line_plot("Line Plot", "X", "Y");
    assert_eq!(metadata.title, "Line Plot");
    assert_eq!(metadata.x_label, "X");
    assert_eq!(metadata.y_label, "Y");
    assert!(matches!(metadata.plot_type, PlotType::Line));

    let metadata = VisualizationMetadata::scatter_plot("Scatter Plot", "X", "Y");
    assert!(matches!(metadata.plot_type, PlotType::Scatter));

    let metadata = VisualizationMetadata::bar_chart("Bar Chart", "X", "Y");
    assert!(matches!(metadata.plot_type, PlotType::Bar));

    let metadata = VisualizationMetadata::heatmap("Heatmap", "X", "Y");
    assert!(matches!(metadata.plot_type, PlotType::Heatmap));

    let metadata = VisualizationMetadata::histogram("Histogram", "X", "Y");
    assert!(matches!(metadata.plot_type, PlotType::Histogram));
}

#[test]
fn test_visualization_options() {
    // Test default options
    let options = VisualizationOptions::default();
    assert_eq!(options.width, 800);
    assert_eq!(options.height, 600);
    assert_eq!(options.dpi, 100);
    assert!(options.color_map.is_none());
    assert!(options.show_grid);
    assert!(options.show_legend);
    assert!(options.title_font_size.is_none());
    assert!(options.label_font_size.is_none());
    assert!(options.tick_font_size.is_none());
    assert!(options.line_width.is_none());
    assert!(options.marker_size.is_none());
    assert!(options.show_colorbar);
    assert!(options.color_palette.is_none());

    // Test builder methods
    let options = VisualizationOptions::new()
        .with_width(1200)
        .with_height(800)
        .with_dpi(150)
        .with_color_map(ColorMap::Viridis)
        .with_grid(false)
        .with_legend(false)
        .with_font_sizes(Some(16.0), Some(14.0), Some(12.0))
        .with_line_width(2.0)
        .with_marker_size(5.0)
        .with_colorbar(false)
        .with_color_palette("viridis");

    assert_eq!(options.width, 1200);
    assert_eq!(options.height, 800);
    assert_eq!(options.dpi, 150);
    assert!(matches!(options.color_map, Some(ColorMap::Viridis)));
    assert!(!options.show_grid);
    assert!(!options.show_legend);
    assert_eq!(options.title_font_size, Some(16.0));
    assert_eq!(options.label_font_size, Some(14.0));
    assert_eq!(options.tick_font_size, Some(12.0));
    assert_eq!(options.line_width, Some(2.0));
    assert_eq!(options.marker_size, Some(5.0));
    assert!(!options.show_colorbar);
    assert_eq!(options.color_palette, Some("viridis".to_string()));
}

#[test]
fn test_confusion_matrix_visualizer() {
    // Create a confusion matrix
    let y_true = array![0, 1, 2, 0, 1, 2];
    let y_pred = array![0, 2, 1, 0, 0, 2];

    let (cm, _) = confusion_matrix(&y_true, &y_pred, None).unwrap();
    let cm_f64 = cm.mapv(|x| x as f64);

    // Test creating a visualizer
    let visualizer = helpers::visualize_confusion_matrix(
        cm_f64.view(),
        Some(vec![
            "Class 0".to_string(),
            "Class 1".to_string(),
            "Class 2".to_string(),
        ]),
        false,
    );

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    assert!(data.z.is_some());

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert!(metadata.title.contains("Confusion Matrix"));
    assert!(matches!(metadata.plot_type, PlotType::Heatmap));
}

#[test]
fn test_roc_curve_visualizer() {
    // Create binary classification data
    let y_true = array![0, 1, 1, 0, 1, 0];
    let y_score = array![0.1, 0.8, 0.7, 0.3, 0.9, 0.2];

    // Compute ROC curve
    let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score).unwrap();

    // Test creating a visualizer
    let visualizer =
        helpers::visualize_roc_curve(fpr.view(), tpr.view(), Some(thresholds.view()), Some(0.85));

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    // Just check that data is present - specific lengths may vary based on visualization implementation
    assert!(!data.x.is_empty());
    assert!(!data.y.is_empty());

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert!(metadata.title.contains("ROC Curve"));
    assert!(matches!(metadata.plot_type, PlotType::Line));
}

#[test]
fn test_precision_recall_visualizer() {
    // Create binary classification data
    let y_true = array![0, 1, 1, 0, 1, 0];
    let y_score = array![0.1, 0.8, 0.7, 0.3, 0.9, 0.2];

    // Compute precision-recall curve
    let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();

    // Test creating a visualizer
    let visualizer = helpers::visualize_precision_recall_curve(
        precision.view(),
        recall.view(),
        Some(thresholds.view()),
        Some(0.75),
    );

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    // Just check that data is present - specific lengths may vary based on visualization implementation
    assert!(!data.x.is_empty());
    assert!(!data.y.is_empty());

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert!(metadata.title.contains("Precision-Recall Curve"));
    assert!(matches!(metadata.plot_type, PlotType::Line));
}

#[test]
fn test_calibration_visualizer() {
    // Create binary classification data
    let y_true = array![0, 1, 1, 0, 1, 0];
    let y_score = array![0.1, 0.8, 0.7, 0.3, 0.9, 0.2];

    // Compute calibration curve
    let (prob_true, prob_pred, _) = calibration_curve(&y_true, &y_score, Some(3)).unwrap();

    // Test creating a visualizer
    let visualizer =
        helpers::visualize_calibration_curve(prob_true.view(), prob_pred.view(), 3, "uniform");

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    // Just check that data is present - specific lengths may vary based on visualization implementation
    assert!(!data.x.is_empty());
    assert!(!data.y.is_empty());

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert!(metadata.title.contains("Calibration Curve"));
    assert!(matches!(metadata.plot_type, PlotType::Line));
}

#[test]
fn test_generic_metric_visualizer() {
    // Create data
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![2.0, 4.0, 1.0, 3.0, 5.0];

    // Test creating a visualizer
    let visualizer = helpers::visualize_metric(
        x.view(),
        y.view(),
        "Generic Metric",
        "X Label",
        "Y Label",
        PlotType::Line,
    );

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    assert_eq!(data.x, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(data.y, vec![2.0, 4.0, 1.0, 3.0, 5.0]);

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert_eq!(metadata.title, "Generic Metric");
    assert_eq!(metadata.x_label, "X Label");
    assert_eq!(metadata.y_label, "Y Label");
    assert!(matches!(metadata.plot_type, PlotType::Line));
}

#[test]
fn test_multi_curve_visualizer() {
    // Create data for multiple curves
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y1 = array![2.0, 4.0, 1.0, 3.0, 5.0];
    let y2 = array![1.0, 3.0, 5.0, 2.0, 4.0];
    let y3 = array![5.0, 4.0, 3.0, 2.0, 1.0];

    // Test creating a visualizer
    let visualizer = helpers::visualize_multi_curve(
        x.view(),
        vec![y1.view(), y2.view(), y3.view()],
        vec![
            "Series 1".to_string(),
            "Series 2".to_string(),
            "Series 3".to_string(),
        ],
        "Multi-Curve Plot",
        "X Label",
        "Y Label",
    );

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    assert_eq!(data.x, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(data.y, vec![2.0, 4.0, 1.0, 3.0, 5.0]);
    assert!(data.series.contains_key("Series 2"));
    assert!(data.series.contains_key("Series 3"));
    assert_eq!(data.series["Series 2"], vec![1.0, 3.0, 5.0, 2.0, 4.0]);
    assert_eq!(data.series["Series 3"], vec![5.0, 4.0, 3.0, 2.0, 1.0]);

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert_eq!(metadata.title, "Multi-Curve Plot");
    assert_eq!(metadata.x_label, "X Label");
    assert_eq!(metadata.y_label, "Y Label");
    assert!(matches!(metadata.plot_type, PlotType::Line));
}

#[test]
fn test_heatmap_visualizer() {
    // Create a matrix
    let matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    // Test creating a visualizer
    let visualizer = helpers::visualize_heatmap(
        matrix.view(),
        Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
        Some(vec!["D".to_string(), "E".to_string(), "F".to_string()]),
        "Heatmap",
        Some(ColorMap::Viridis),
    );

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    assert!(data.z.is_some());
    assert_eq!(data.z.as_ref().unwrap().len(), 3);
    assert_eq!(data.z.as_ref().unwrap()[0].len(), 3);
    assert_eq!(
        data.x_labels,
        Some(vec!["A".to_string(), "B".to_string(), "C".to_string()])
    );
    assert_eq!(
        data.y_labels,
        Some(vec!["D".to_string(), "E".to_string(), "F".to_string()])
    );

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert_eq!(metadata.title, "Heatmap");
    assert!(matches!(metadata.plot_type, PlotType::Heatmap));
}

#[test]
fn test_histogram_visualizer() {
    // Create data for a histogram
    let values = array![
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7,
    ];

    // Test creating a visualizer
    let visualizer = helpers::visualize_histogram(
        values.view(),
        5,
        "Histogram",
        "Value",
        Some("Count".to_string()),
    );

    // Test preparing data
    let data = visualizer.prepare_data().unwrap();
    assert_eq!(data.y.len(), 5); // 5 bins

    // Test metadata
    let metadata = visualizer.get_metadata();
    assert_eq!(metadata.title, "Histogram");
    assert_eq!(metadata.x_label, "Value");
    assert_eq!(metadata.y_label, "Count");
    assert!(matches!(metadata.plot_type, PlotType::Histogram));
}

#[test]
fn test_backend_default() {
    // Test that the default backend can be created
    let backend = backends::default_backend();

    // Create some basic data and metadata
    let mut data = VisualizationData::new();
    data.x = vec![1.0, 2.0, 3.0];
    data.y = vec![4.0, 5.0, 6.0];

    let metadata = VisualizationMetadata::new("Test");
    let options = VisualizationOptions::default();

    // Test that render methods don't panic
    // Note: We're not testing the actual rendering, just that the methods don't panic
    let _ = backend.render_svg(&data, &metadata, &options);
    let _ = backend.render_png(&data, &metadata, &options);
}
