using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class InferencePlaygroundViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    private readonly OnnxModel _model;
    
    [ObservableProperty]
    private ObservableCollection<InferenceInput> _inputs = new();
    
    [ObservableProperty]
    private InferenceResult? _result;
    
    [ObservableProperty]
    private bool _isRunning = false;
    
    [ObservableProperty]
    private string _runButtonText = "Run Inference";
    
    [ObservableProperty]
    private string _statusMessage = "Ready";
    
    [ObservableProperty]
    private int _batchSize = 1;
    
    public InferencePlaygroundViewModel(MainViewModel mainViewModel, OnnxModel model)
    {
        _mainViewModel = mainViewModel;
        _model = model;
        Title = "ONNX Studio - " + model.Name + " Inference";
        
        // Create mock inputs based on model type
        Inputs = new ObservableCollection<InferenceInput>(MockDataGenerator.CreateMockInferenceInputs(model));
    }
    
    [RelayCommand]
    private void Back()
    {
        _mainViewModel.ShowDashboard();
    }
    
    [RelayCommand]
    private void SwitchToInspector()
    {
        _mainViewModel.ShowModelInspector(_model);
    }
    
    [RelayCommand]
    private void SwitchToApi()
    {
        _mainViewModel.ShowApiConfig(_model);
    }
    
    [RelayCommand]
    private async Task RunInferenceAsync()
    {
        if (IsRunning) return;
        
        IsRunning = true;
        RunButtonText = "Running...⠋";
        StatusMessage = "Running inference...";
        Result = null;
        
        // Simulate inference delay
        await Task.Delay(1500);
        
        // Create mock result
        Result = MockDataGenerator.CreateMockInferenceResult(_model, Inputs.ToList());
        StatusMessage = "Inference completed in " + Result.ExecutionTimeMs + "ms";
        
        IsRunning = false;
        RunButtonText = "Run Inference";
    }
    
    [RelayCommand]
    private void CopyResults()
    {
        if (Result != null)
        {
            StatusMessage = "Results copied to clipboard";
        }
    }
    
    [RelayCommand]
    private void SaveResults()
    {
        if (Result != null)
        {
            StatusMessage = "Results saved to file";
        }
    }
    
    public void UpdateInputValue(InferenceInput input, object value)
    {
        input.Value = value;
        // Validate input
        ValidateInput(input);
    }
    
    private void ValidateInput(InferenceInput input)
    {
        // Simple validation for demo
        input.IsValid = true;
        input.ErrorMessage = null;
        
        if (input.IsRequired && input.Value == null)
        {
            input.IsValid = false;
            input.ErrorMessage = "This field is required";
        }
        else if (input.InputType == "Vector" && input.VectorValues != null)
        {
            var values = input.VectorValues.Split(new[] { ',', ' ', ';' }, StringSplitOptions.RemoveEmptyEntries);
            if (values.Length != input.Shape.Aggregate(1, (a, b) => a * (int)b))
            {
                input.IsValid = false;
                input.ErrorMessage = "Incorrect number of values";
            }
        }
    }
    
    public string GetInputShapeDisplay(InferenceInput input)
    {
        return string.Join("x", input.Shape);
    }
    
    public override void OnLoaded()
    {
        base.OnLoaded();
        StatusMessage = "Ready to run inference";
        
        // Pre-validate all inputs
        foreach (var input in Inputs)
        {
            ValidateInput(input);
        }
    }
}
