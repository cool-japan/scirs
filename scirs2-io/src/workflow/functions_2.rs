//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{IoError, Result};
use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::{HashMap, HashSet};

use super::types::{Task, Workflow, WorkflowStatus};

/// External workflow engine integration
pub mod engines {
    use super::*;
    /// Trait for external workflow engine adapters
    pub trait WorkflowEngineAdapter: Send + Sync {
        /// Convert internal workflow to engine-specific format
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String>;
        /// Import workflow from engine-specific format
        fn importworkflow(&self, definition: &str) -> Result<Workflow>;
        /// Submit workflow for execution
        fn submit(&self, workflow: &Workflow) -> Result<String>;
        /// Get execution status
        fn get_status(&self, executionid: &str) -> Result<WorkflowStatus>;
        /// Cancel execution
        fn cancel(&self, executionid: &str) -> Result<()>;
    }
    /// Apache Airflow adapter
    pub struct AirflowAdapter {
        api_url: String,
        auth_token: Option<String>,
    }
    impl AirflowAdapter {
        pub fn new(api_url: impl Into<String>) -> Self {
            Self {
                api_url: api_url.into(),
                auth_token: None,
            }
        }
        pub fn with_auth(mut self, token: impl Into<String>) -> Self {
            self.auth_token = Some(token.into());
            self
        }
    }
    impl WorkflowEngineAdapter for AirflowAdapter {
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String> {
            let mut dag_code = String::new();
            dag_code.push_str("from airflow import DAG\n");
            dag_code.push_str("from airflow.operators.python import PythonOperator\n");
            dag_code.push_str("from datetime import datetime, timedelta\n\n");
            dag_code.push_str("dag = DAG(\n");
            dag_code.push_str(&format!("    '{}',\n", workflow.id));
            dag_code.push_str(&format!(
                "    description='{}',\n",
                workflow.description.as_deref().unwrap_or("")
            ));
            dag_code.push_str("    default_args={\n");
            dag_code.push_str("        'owner': 'scirs2',\n");
            dag_code.push_str("        'retries': 3,\n");
            dag_code.push_str("        'retry_delay': timedelta(minutes=5),\n");
            dag_code.push_str("    },\n");
            dag_code.push_str("    schedule_interval=None,\n");
            dag_code.push_str("    start_date=datetime(2024, 1, 1),\n");
            dag_code.push_str("    catchup=False,\n");
            dag_code.push_str(")\n\n");
            for task in &workflow.tasks {
                dag_code.push_str(&format!("{} = PythonOperator(\n", task.id));
                dag_code.push_str(&format!("    task_id='{}',\n", task.id));
                dag_code.push_str(&format!(
                    "    python_callable=lambda: print('{}'),\n",
                    task.name
                ));
                dag_code.push_str("    dag=dag,\n");
                dag_code.push_str(")\n\n");
            }
            for (task_id, deps) in &workflow.dependencies {
                for dep in deps {
                    dag_code.push_str(&format!("{dep} >> {task_id}\n"));
                }
            }
            Ok(dag_code)
        }
        fn importworkflow(&self, definition: &str) -> Result<Workflow> {
            // Round-trip parser for the Airflow DAG format produced by exportworkflow.
            // Extracts: DAG id, task ids + names, and dependency edges (dep >> task_id).
            use super::super::types::{ResourceRequirements, TaskType, WorkflowConfig};
            use crate::metadata::Metadata;

            let mut workflow_id = String::from("imported_dag");
            let mut tasks = Vec::new();
            let mut dependencies: HashMap<String, Vec<String>> = HashMap::new();
            // State tracking for multi-line DAG(...) constructor
            let mut inside_dag_constructor = false;
            let mut dag_id_found = false;

            for line in definition.lines() {
                let trimmed = line.trim();

                // The exporter emits:
                //   dag = DAG(
                //       'dag_id',
                // Handle both single-line `dag = DAG('id', ...)` and multi-line forms.
                if trimmed == "dag = DAG(" {
                    inside_dag_constructor = true;
                    continue;
                }
                // Single-line form: dag = DAG('dag_id', ...)
                if trimmed.starts_with("dag = DAG('") {
                    if let Some(inner) = trimmed.strip_prefix("dag = DAG('") {
                        if let Some(end) = inner.find('\'') {
                            workflow_id = inner[..end].to_string();
                            dag_id_found = true;
                        }
                    }
                }
                // Inside multi-line DAG constructor: first quoted string is the dag id
                if inside_dag_constructor && !dag_id_found {
                    if trimmed.starts_with('\'') {
                        let inner = trimmed.trim_start_matches('\'');
                        if let Some(end) = inner.find('\'') {
                            workflow_id = inner[..end].to_string();
                            dag_id_found = true;
                        }
                    }
                    // Detect end of constructor
                    if trimmed == ")" {
                        inside_dag_constructor = false;
                    }
                }

                // Extract task: task_id = PythonOperator(
                // followed by:  task_id='task_id', then python_callable=lambda: print('task_name')
                if trimmed.contains(" = PythonOperator(") {
                    if let Some(eq_pos) = trimmed.find(" = PythonOperator(") {
                        let task_id = trimmed[..eq_pos].trim().to_string();
                        if !task_id.is_empty() {
                            tasks.push((task_id, String::new()));
                        }
                    }
                }

                // Extract task name from: python_callable=lambda: print('task_name'),
                if trimmed.starts_with("python_callable=lambda: print('") {
                    if let Some(stripped) = trimmed.strip_prefix("python_callable=lambda: print('")
                    {
                        if let Some(end) = stripped.find('\'') {
                            let task_name = stripped[..end].to_string();
                            if let Some(last) = tasks.last_mut() {
                                last.1 = task_name;
                            }
                        }
                    }
                }

                // Extract dependency: dep >> task_id
                if trimmed.contains(" >> ") && !trimmed.starts_with('#') {
                    let parts: Vec<&str> = trimmed.split(" >> ").collect();
                    if parts.len() == 2 {
                        let dep = parts[0].trim().to_string();
                        let task_id = parts[1].trim().to_string();
                        if !dep.is_empty() && !task_id.is_empty() {
                            dependencies.entry(task_id).or_default().push(dep);
                        }
                    }
                }
            }

            let task_objects: Vec<Task> = tasks
                .into_iter()
                .map(|(id, name)| {
                    let display_name = if name.is_empty() { id.clone() } else { name };
                    Task {
                        id,
                        name: display_name,
                        task_type: TaskType::Script,
                        config: serde_json::Value::Null,
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        resources: ResourceRequirements::default(),
                    }
                })
                .collect();

            Ok(Workflow {
                id: workflow_id.clone(),
                name: workflow_id,
                description: None,
                tasks: task_objects,
                dependencies,
                config: WorkflowConfig::default(),
                metadata: Metadata::new(),
            })
        }
        fn submit(&self, workflow: &Workflow) -> Result<String> {
            let executionid = format!("{}_run_{}", workflow.id, Utc::now().timestamp());
            Ok(executionid)
        }
        fn get_status(&self, _executionid: &str) -> Result<WorkflowStatus> {
            Ok(WorkflowStatus::Running)
        }
        fn cancel(&self, _executionid: &str) -> Result<()> {
            Ok(())
        }
    }
    /// Prefect adapter
    pub struct PrefectAdapter {
        api_url: String,
        project_name: String,
    }
    impl PrefectAdapter {
        pub fn new(api_url: impl Into<String>, project: impl Into<String>) -> Self {
            Self {
                api_url: api_url.into(),
                project_name: project.into(),
            }
        }
    }
    impl WorkflowEngineAdapter for PrefectAdapter {
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String> {
            let mut flow_code = String::new();
            flow_code.push_str("from prefect import flow, task\n");
            flow_code.push_str("from prefect.task_runners import SequentialTaskRunner\n\n");
            for task in &workflow.tasks {
                flow_code.push_str(&format!("@task(name='{}')\n", task.name));
                flow_code.push_str(&format!("def {}():\n", task.id));
                flow_code.push_str(&format!("    print('Executing {}')\n", task.name));
                flow_code.push_str("    return True\n\n");
            }
            flow_code.push_str(&format!(
                "@flow(name='{}', task_runner=SequentialTaskRunner())\n",
                workflow.name
            ));
            flow_code.push_str("def workflow_flow():\n");
            let mut executed = HashSet::new();
            let mut to_execute: Vec<_> = workflow.tasks.iter().map(|t| &t.id).collect();
            while !to_execute.is_empty() {
                let mut progress = false;
                to_execute.retain(|task_id| {
                    let deps = workflow.dependencies.get(*task_id);
                    let can_execute =
                        deps.is_none_or(|d| d.iter().all(|dep| executed.contains(dep)));
                    if can_execute {
                        flow_code.push_str(&format!("    {task_id}()\n"));
                        executed.insert((*task_id).clone());
                        progress = true;
                        false
                    } else {
                        true
                    }
                });
                if !progress && !to_execute.is_empty() {
                    return Err(IoError::Other("Circular dependency detected".to_string()));
                }
            }
            flow_code.push_str("\nif __name__ == '__main__':\n");
            flow_code.push_str("    workflow_flow()\n");
            Ok(flow_code)
        }
        fn importworkflow(&self, definition: &str) -> Result<Workflow> {
            // Round-trip parser for the Prefect flow format produced by exportworkflow.
            // Extracts: flow name, task ids + names, and call order from workflow_flow().
            use super::super::types::{ResourceRequirements, TaskType, WorkflowConfig};
            use crate::metadata::Metadata;

            let mut workflow_name = String::from("imported_flow");
            let mut workflow_id = String::from("imported_flow");
            // Map from task function id -> task name (from @task(name='...'))
            let mut pending_task_name: Option<String> = None;
            let mut task_name_map: HashMap<String, String> = HashMap::new();
            // Task call order extracted from workflow_flow()
            let mut ordered_task_ids: Vec<String> = Vec::new();
            let mut inside_flow_body = false;

            for line in definition.lines() {
                let trimmed = line.trim();

                // @flow(name='workflow_name', ...)
                if trimmed.starts_with("@flow(name='") {
                    if let Some(stripped) = trimmed.strip_prefix("@flow(name='") {
                        if let Some(end) = stripped.find('\'') {
                            workflow_name = stripped[..end].to_string();
                            workflow_id = workflow_name.replace(' ', "_").to_lowercase();
                        }
                    }
                }

                // @task(name='task_name')
                if trimmed.starts_with("@task(name='") {
                    if let Some(stripped) = trimmed.strip_prefix("@task(name='") {
                        if let Some(end) = stripped.find('\'') {
                            pending_task_name = Some(stripped[..end].to_string());
                        }
                    }
                }

                // def task_id():  — task function definition
                if trimmed.starts_with("def ")
                    && trimmed.ends_with("():")
                    && !trimmed.starts_with("def workflow_flow")
                {
                    if let Some(stripped) = trimmed.strip_prefix("def ") {
                        if let Some(end) = stripped.find("():") {
                            let task_id = stripped[..end].to_string();
                            let name = pending_task_name.take().unwrap_or_else(|| task_id.clone());
                            task_name_map.insert(task_id, name);
                        }
                    }
                }

                // def workflow_flow(): — start of flow body
                if trimmed == "def workflow_flow():" {
                    inside_flow_body = true;
                    continue;
                }

                // Inside flow body: lines like "    task_id()"
                if inside_flow_body {
                    if trimmed.starts_with("if __name__") {
                        inside_flow_body = false;
                    } else if trimmed.ends_with("()") {
                        let task_id = trimmed.trim_end_matches("()").to_string();
                        if !task_id.is_empty() && task_name_map.contains_key(&task_id) {
                            ordered_task_ids.push(task_id);
                        }
                    }
                }
            }

            let tasks: Vec<Task> = ordered_task_ids
                .iter()
                .map(|id| {
                    let name = task_name_map.get(id).cloned().unwrap_or_else(|| id.clone());
                    Task {
                        id: id.clone(),
                        name,
                        task_type: TaskType::Script,
                        config: serde_json::Value::Null,
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        resources: ResourceRequirements::default(),
                    }
                })
                .collect();

            // Prefect exporter serialises tasks in topological order inline (no explicit deps).
            // The ordering itself encodes sequencing but we cannot recover edge data without
            // more information — return empty dependency map as the safe round-trip value.
            Ok(Workflow {
                id: workflow_id,
                name: workflow_name,
                description: None,
                tasks,
                dependencies: HashMap::new(),
                config: WorkflowConfig::default(),
                metadata: Metadata::new(),
            })
        }
        fn submit(&self, workflow: &Workflow) -> Result<String> {
            let flow_run_id = uuid::Uuid::new_v4().to_string();
            Ok(flow_run_id)
        }
        fn get_status(&self, _executionid: &str) -> Result<WorkflowStatus> {
            Ok(WorkflowStatus::Running)
        }
        fn cancel(&self, _executionid: &str) -> Result<()> {
            Ok(())
        }
    }
    /// Dagster adapter
    pub struct DagsterAdapter {
        repository_url: String,
    }
    impl DagsterAdapter {
        /// Create a new Dagster adapter
        pub fn new(repository_url: impl Into<String>) -> Self {
            Self {
                repository_url: repository_url.into(),
            }
        }
    }
    impl WorkflowEngineAdapter for DagsterAdapter {
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String> {
            let mut job_code = String::new();
            job_code.push_str("from dagster import job, op, Config\n\n");
            for task in &workflow.tasks {
                job_code.push_str(&format!("@op(name='{}')\n", task.id));
                job_code.push_str(&format!("def {}(context):\n", task.id));
                job_code.push_str(&format!(
                    "    context.log.info('Executing {}')\n",
                    task.name
                ));
                job_code.push_str("    return True\n\n");
            }
            job_code.push_str(&format!("@job(name='{}')\n", workflow.id));
            job_code.push_str("def workflow_job():\n");
            for task in &workflow.tasks {
                if let Some(deps) = workflow.dependencies.get(&task.id) {
                    let deps_str = deps.join(", ");
                    job_code.push_str(&format!("    {}({}())\n", task.id, deps_str));
                } else {
                    job_code.push_str(&format!("    {}()\n", task.id));
                }
            }
            Ok(job_code)
        }
        fn importworkflow(&self, definition: &str) -> Result<Workflow> {
            // Round-trip parser for the Dagster job format produced by exportworkflow.
            // Extracts: job name (workflow id), op names, and call-graph deps from workflow_job().
            use super::super::types::{ResourceRequirements, TaskType, WorkflowConfig};
            use crate::metadata::Metadata;

            let mut workflow_id = String::from("imported_job");
            let mut pending_op_id: Option<String> = None;
            // Map task_id -> task_name from @op(name='...') + context.log.info line
            let mut task_info: Vec<(String, String)> = Vec::new();
            let mut dependencies: HashMap<String, Vec<String>> = HashMap::new();
            let mut inside_job_body = false;

            for line in definition.lines() {
                let trimmed = line.trim();

                // @job(name='job_id')
                if trimmed.starts_with("@job(name='") {
                    if let Some(stripped) = trimmed.strip_prefix("@job(name='") {
                        if let Some(end) = stripped.find('\'') {
                            workflow_id = stripped[..end].to_string();
                        }
                    }
                }

                // @op(name='task_id')
                if trimmed.starts_with("@op(name='") {
                    if let Some(stripped) = trimmed.strip_prefix("@op(name='") {
                        if let Some(end) = stripped.find('\'') {
                            pending_op_id = Some(stripped[..end].to_string());
                        }
                    }
                }

                // context.log.info('Executing task_name')
                if trimmed.starts_with("context.log.info('Executing ") {
                    if let Some(stripped) = trimmed.strip_prefix("context.log.info('Executing ") {
                        if let Some(end) = stripped.find('\'') {
                            let task_name = stripped[..end].to_string();
                            if let Some(op_id) = pending_op_id.take() {
                                task_info.push((op_id, task_name));
                            }
                        }
                    }
                }

                // def workflow_job():
                if trimmed == "def workflow_job():" {
                    inside_job_body = true;
                    continue;
                }

                // Inside job body: lines like "    task_id(dep1(), dep2())" or "    task_id()"
                if inside_job_body && trimmed.starts_with(|c: char| c.is_alphabetic() || c == '_') {
                    // Parse: task_id(dep_calls) where dep_calls may be empty or "dep1(), dep2()"
                    if let Some(paren_pos) = trimmed.find('(') {
                        let task_id = trimmed[..paren_pos].trim().to_string();
                        let rest = &trimmed[paren_pos + 1..];
                        // Remove trailing ')'
                        let args = rest.trim_end_matches(')').trim();
                        if !args.is_empty() {
                            // Each dep looks like "dep_id()" — extract dep_id
                            let deps: Vec<String> = args
                                .split(',')
                                .filter_map(|seg| {
                                    let seg = seg.trim();
                                    // dep calls look like "dep_id()"
                                    seg.find('(').map(|p| seg[..p].trim().to_string())
                                })
                                .filter(|s| !s.is_empty())
                                .collect();
                            if !deps.is_empty() {
                                dependencies.insert(task_id, deps);
                            }
                        }
                    }
                }
            }

            let tasks: Vec<Task> = task_info
                .into_iter()
                .map(|(id, name)| Task {
                    id,
                    name,
                    task_type: TaskType::Script,
                    config: serde_json::Value::Null,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    resources: ResourceRequirements::default(),
                })
                .collect();

            Ok(Workflow {
                id: workflow_id.clone(),
                name: workflow_id,
                description: None,
                tasks,
                dependencies,
                config: WorkflowConfig::default(),
                metadata: Metadata::new(),
            })
        }
        fn submit(&self, workflow: &Workflow) -> Result<String> {
            Ok(uuid::Uuid::new_v4().to_string())
        }
        fn get_status(&self, _executionid: &str) -> Result<WorkflowStatus> {
            Ok(WorkflowStatus::Running)
        }
        fn cancel(&self, _executionid: &str) -> Result<()> {
            Ok(())
        }
    }
}
/// Dynamic workflow generation
pub mod dynamic {
    use super::*;
    /// Dynamic workflow generator
    pub struct DynamicWorkflowGenerator {
        templates: HashMap<String, WorkflowTemplate>,
    }
    #[derive(Debug, Clone)]
    pub struct WorkflowTemplate {
        pub baseworkflow: Workflow,
        pub parameters: Vec<ParameterDef>,
        pub generators: Vec<TaskGenerator>,
    }
    #[derive(Debug, Clone)]
    pub struct ParameterDef {
        pub name: String,
        pub param_type: ParameterType,
        pub required: bool,
        pub default: Option<serde_json::Value>,
    }
    #[derive(Debug, Clone)]
    pub enum ParameterType {
        String,
        Integer,
        Float,
        Boolean,
        List(Box<ParameterType>),
        Object,
    }
    #[derive(Debug, Clone)]
    pub enum TaskGenerator {
        ForEach {
            parameter: String,
            task_template: Task,
        },
        Conditional {
            condition: String,
            true_tasks: Vec<Task>,
            false_tasks: Vec<Task>,
        },
        Repeat {
            count_param: String,
            task_template: Task,
        },
    }
    impl Default for DynamicWorkflowGenerator {
        fn default() -> Self {
            Self::new()
        }
    }
    impl DynamicWorkflowGenerator {
        pub fn new() -> Self {
            Self {
                templates: HashMap::new(),
            }
        }
        /// Register a workflow template
        pub fn register_template(&mut self, name: impl Into<String>, template: WorkflowTemplate) {
            self.templates.insert(name.into(), template);
        }
        /// Generate workflow from template
        pub fn generate(
            &self,
            template_name: &str,
            params: HashMap<String, serde_json::Value>,
        ) -> Result<Workflow> {
            let template = self.templates.get(template_name).ok_or_else(|| {
                IoError::NotFound(format!("Template '{template_name}' not found"))
            })?;
            for param_def in &template.parameters {
                if param_def.required && !params.contains_key(&param_def.name) {
                    return Err(IoError::ValidationError(format!(
                        "Required parameter '{}' not provided",
                        param_def.name
                    )));
                }
            }
            let mut workflow = template.baseworkflow.clone();
            workflow.id = format!("{}_{}", workflow.id, Utc::now().timestamp());
            for generator in &template.generators {
                self.apply_generator(&mut workflow, generator, &params)?;
            }
            Ok(workflow)
        }
        fn apply_generator(
            &self,
            workflow: &mut Workflow,
            generator: &TaskGenerator,
            params: &HashMap<String, serde_json::Value>,
        ) -> Result<()> {
            match generator {
                TaskGenerator::ForEach {
                    parameter,
                    task_template,
                } => {
                    if let Some(serde_json::Value::Array(items)) = params.get(parameter) {
                        for (i, item) in items.iter().enumerate() {
                            let mut task = task_template.clone();
                            task.id = format!("{}_{}", task.id, i);
                            task.name = format!("{} [{}]", task.name, i);
                            if let serde_json::Value::Object(mut config) = task.config.clone() {
                                config.insert("item".to_string(), item.clone());
                                task.config = serde_json::Value::Object(config);
                            }
                            workflow.tasks.push(task);
                        }
                    }
                }
                TaskGenerator::Conditional {
                    condition,
                    true_tasks,
                    false_tasks,
                } => {
                    let condition_result = self.evaluate_condition(condition, params)?;
                    if condition_result {
                        workflow.tasks.extend(true_tasks.iter().cloned());
                    } else {
                        workflow.tasks.extend(false_tasks.iter().cloned());
                    }
                }
                TaskGenerator::Repeat {
                    count_param,
                    task_template,
                } => {
                    if let Some(serde_json::Value::Number(n)) = params.get(count_param) {
                        if let Some(count) = n.as_u64() {
                            for i in 0..count {
                                let mut task = task_template.clone();
                                task.id = format!("{}_{}", task.id, i);
                                task.name = format!("{} [{}]", task.name, i);
                                workflow.tasks.push(task);
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        fn evaluate_condition(
            &self,
            condition: &str,
            params: &HashMap<String, serde_json::Value>,
        ) -> Result<bool> {
            if let Some((param, value)) = condition.split_once("==") {
                let param = param.trim();
                let value = value.trim().trim_matches('"');
                if let Some(serde_json::Value::String(s)) = params.get(param) {
                    return Ok(s == value);
                }
            }
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{ResourceRequirements, Task, TaskType, Workflow, WorkflowConfig};
    use super::*;
    use crate::metadata::Metadata;

    fn make_workflow(id: &str, tasks: Vec<Task>) -> Workflow {
        Workflow {
            id: id.to_string(),
            name: id.to_string(),
            description: None,
            tasks,
            dependencies: HashMap::new(),
            config: WorkflowConfig::default(),
            metadata: Metadata::new(),
        }
    }

    fn make_task(id: &str, name: &str) -> Task {
        Task {
            id: id.to_string(),
            name: name.to_string(),
            task_type: TaskType::Script,
            config: serde_json::Value::Null,
            inputs: Vec::new(),
            outputs: Vec::new(),
            resources: ResourceRequirements::default(),
        }
    }

    #[test]
    fn test_airflow_import_roundtrip() {
        use engines::{AirflowAdapter, WorkflowEngineAdapter};
        let adapter = AirflowAdapter::new("http://localhost:8080");
        let tasks = vec![
            make_task("ingest", "Ingest Data"),
            make_task("transform", "Transform Data"),
        ];
        let workflow = make_workflow("my_pipeline", tasks);
        let exported = adapter
            .exportworkflow(&workflow)
            .expect("Export should succeed");
        let imported = adapter
            .importworkflow(&exported)
            .expect("Import should succeed");
        assert_eq!(imported.id, "my_pipeline");
        assert_eq!(imported.tasks.len(), 2);
        let ids: Vec<&str> = imported.tasks.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"ingest"), "Expected 'ingest' in {:?}", ids);
        assert!(
            ids.contains(&"transform"),
            "Expected 'transform' in {:?}",
            ids
        );
    }

    #[test]
    fn test_prefect_import_roundtrip() {
        use engines::{PrefectAdapter, WorkflowEngineAdapter};
        let adapter = PrefectAdapter::new("http://localhost:4200", "test_project");
        let tasks = vec![
            make_task("fetch", "Fetch Records"),
            make_task("validate", "Validate Records"),
        ];
        let workflow = make_workflow("data_flow", tasks);
        let exported = adapter
            .exportworkflow(&workflow)
            .expect("Export should succeed");
        let imported = adapter
            .importworkflow(&exported)
            .expect("Import should succeed");
        assert!(imported.tasks.len() >= 1, "Should have at least one task");
    }

    #[test]
    fn test_dagster_import_roundtrip() {
        use engines::{DagsterAdapter, WorkflowEngineAdapter};
        let adapter = DagsterAdapter::new("");
        let tasks = vec![
            make_task("load", "Load Source"),
            make_task("process", "Process Data"),
        ];
        let mut workflow = make_workflow("etl_job", tasks);
        workflow
            .dependencies
            .insert("process".to_string(), vec!["load".to_string()]);
        let exported = adapter
            .exportworkflow(&workflow)
            .expect("Export should succeed");
        let imported = adapter
            .importworkflow(&exported)
            .expect("Import should succeed");
        assert_eq!(imported.id, "etl_job");
        assert!(imported.tasks.len() >= 2, "Should have at least 2 tasks");
        // Dependency: process depends on load
        if let Some(deps) = imported.dependencies.get("process") {
            assert!(deps.contains(&"load".to_string()));
        }
    }
}
