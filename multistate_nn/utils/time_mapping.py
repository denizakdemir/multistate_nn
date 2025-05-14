"""Time mapping utilities for MultiStateNN models."""

from typing import Dict, Union, Optional, List, Sequence, Tuple, Any
import numpy as np
import pandas as pd
import torch
import warnings


class TimeMapper:
    """Maps between original time values and internal time indices.
    
    This class provides functionality to convert between original time values 
    in the data and the internal integer time indices used by the model.
    This allows the model to work with arbitrary time scales while maintaining
    its internal discrete-time representation.
    
    Parameters
    ----------
    time_values : Union[Sequence, np.ndarray, pd.Series]
        Original time values from the dataset
    handle_censoring : bool, optional
        Whether to specially handle censored observations when mapping times
    
    Attributes
    ----------
    time_values : np.ndarray
        Sorted unique time values from the dataset
    time_to_idx : Dict[float, int]
        Dictionary mapping time values to indices
    idx_to_time : Dict[int, float]
        Dictionary mapping indices to time values
    handle_censoring : bool
        Whether to handle censored observations specially
    """
    
    def __init__(self, 
                time_values: Union[Sequence, np.ndarray, pd.Series],
                handle_censoring: bool = True):
        """Initialize time mapper with original time values."""
        # Get unique time values and sort them
        unique_times = np.sort(np.unique(time_values))
        self.time_values = unique_times
        
        # Create mappings
        self.time_to_idx = {float(t): i for i, t in enumerate(self.time_values)}
        self.idx_to_time = {i: float(t) for i, t in enumerate(self.time_values)}
        
        # Store the number of time points
        self.n_time_points = len(self.time_values)
        
        # Whether to specially handle censored observations
        self.handle_censoring = handle_censoring
    
    def to_idx(self, time: Union[float, Sequence, np.ndarray, pd.Series]) -> Union[int, np.ndarray]:
        """Convert original time to index.
        
        Parameters
        ----------
        time : Union[float, Sequence, np.ndarray, pd.Series]
            Original time value(s) to convert
            
        Returns
        -------
        Union[int, np.ndarray]
            Index or array of indices corresponding to the input time(s)
        """
        if isinstance(time, (int, float)):
            idx = self.time_to_idx.get(float(time))
            if idx is None:
                # For values not in the mapping, use map_continuous_time instead
                return self.map_continuous_time(float(time))
            return idx
        else:
            # Handle arrays/sequences by using a vectorized approach
            # Convert to numpy array if not already
            times = np.asarray(time)
            result = np.zeros(times.shape, dtype=int)
            
            # Use a vectorized approach with a dictionary comprehension
            for i, t in enumerate(times.flatten()):
                idx = self.time_to_idx.get(float(t))
                if idx is None:
                    # For values not in the mapping, use map_continuous_time
                    result.flat[i] = self.map_continuous_time(float(t))
                else:
                    result.flat[i] = idx
                
            return result
    
    def to_time(self, idx: Union[int, Sequence, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert index to original time.
        
        Parameters
        ----------
        idx : Union[int, Sequence, np.ndarray]
            Index or indices to convert
            
        Returns
        -------
        Union[float, np.ndarray]
            Original time value(s) corresponding to the input index/indices
        """
        if isinstance(idx, (int, np.integer)):
            index = int(idx)
            # Clamp index to valid range first
            clamped_index = max(0, min(index, len(self.time_values) - 1))
            
            # Return time from mapping
            if clamped_index < len(self.time_values):
                return self.time_values[clamped_index]
            # Fallback - should never reach here after clamping
            return self.time_values[-1]
        else:
            # Handle arrays of indices
            indices = np.asarray(idx)
            result = np.zeros(indices.shape, dtype=float)
            
            # For vectorized operation, we clamp all indices to valid range
            clamped_indices = np.clip(indices, 0, len(self.time_values) - 1)
            
            # Then convert to time values
            for i, index in enumerate(clamped_indices.flatten()):
                idx = int(index)
                result.flat[i] = self.time_values[idx]
                
            return result
    
    def map_df_time_to_idx(self, df: pd.DataFrame, time_col: str = 'time', 
                          idx_col: str = 'time_idx', 
                          censoring_col: Optional[str] = None) -> pd.DataFrame:
        """Map time column in DataFrame to indices, with optional censoring handling.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time values
        time_col : str, optional
            Name of the column containing time values
        idx_col : str, optional
            Name of the column to store indices
        censoring_col : Optional[str], optional
            Name of the column containing censoring information (1=censored, 0=observed)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added index column
        """
        df = df.copy()
        
        # Check if we need to handle censored observations differently
        if self.handle_censoring and censoring_col is not None and censoring_col in df.columns:
            # For censored observations, we need to think about how they're handled
            # A common approach is to include them up to the observed (censoring) time
            
            # For now, we map all times to indices regardless of censoring status
            # This behavior could be customized based on specific requirements
            df[idx_col] = df[time_col].map(lambda t: self.to_idx(t))
            
            # Flag to indicate that the DataFrame has been processed with censoring awareness
            df['_censoring_handled'] = True
        else:
            # Standard mapping without censoring handling
            df[idx_col] = df[time_col].map(lambda t: self.to_idx(t))
            
        return df
    
    def map_df_idx_to_time(self, df: pd.DataFrame, idx_col: str = 'time_idx',
                          time_col: str = 'time',
                          censoring_col: Optional[str] = None) -> pd.DataFrame:
        """Map index column in DataFrame to original time values, handling censoring.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing indices
        idx_col : str, optional
            Name of the column containing indices
        time_col : str, optional
            Name of the column to store time values
        censoring_col : Optional[str], optional
            Name of the column containing censoring information (1=censored, 0=observed)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added time column
        """
        df = df.copy()
        
        # Basic mapping from indices to time values
        df[time_col] = df[idx_col].map(lambda i: self.to_time(i))
        
        # Check if we should handle censoring specially
        if self.handle_censoring and censoring_col is not None and censoring_col in df.columns:
            # If needed, special handling for censored observations
            # For example, adding an indicator or handling right-censored times differently
            # This is a placeholder for application-specific behaviors
            pass
            
        return df
    
    def get_time_grid(self, n_points: Optional[int] = None) -> np.ndarray:
        """Get evenly spaced time points within the original time range.
        
        Parameters
        ----------
        n_points : Optional[int], optional
            Number of time points to generate. If None, uses all unique time points.
            
        Returns
        -------
        np.ndarray
            Array of evenly spaced time points
        """
        if n_points is None:
            return self.time_values
        
        # Create evenly spaced time points
        min_time = self.time_values[0]
        max_time = self.time_values[-1]
        return np.linspace(min_time, max_time, n_points)
        
    def extend_with_extrapolation(self, max_time: float, n_points: Optional[int] = None) -> 'TimeMapper':
        """Extend the time mapping with extrapolated time points.
        
        Parameters
        ----------
        max_time : float
            Maximum time value to extend to
        n_points : Optional[int], optional
            Number of additional points to add. If None, adds points 
            maintaining the average interval from existing points.
            
        Returns
        -------
        TimeMapper
            New TimeMapper instance with extended time points
        """
        if max_time <= self.time_values[-1]:
            # No need to extend
            return self
            
        # Calculate average step from existing data
        if len(self.time_values) > 1:
            avg_step = (self.time_values[-1] - self.time_values[0]) / (len(self.time_values) - 1)
        else:
            # If only one time point, use 1.0 as default step
            avg_step = 1.0
            
        if n_points is None:
            # Calculate number of points needed to reach max_time with avg_step
            n_additional = int(np.ceil((max_time - self.time_values[-1]) / avg_step))
            
            # Generate extended time values
            extended_times = np.arange(1, n_additional + 1) * avg_step + self.time_values[-1]
        else:
            # Generate evenly spaced points from last existing time to max_time
            extended_times = np.linspace(self.time_values[-1] + avg_step, max_time, n_points)
            
        # Create a new array combining existing and new times
        new_time_values = np.concatenate([self.time_values, extended_times])
        
        # Create new TimeMapper
        return TimeMapper(new_time_values)
    
    def get_closest_idx(self, time: float) -> int:
        """Get the closest index for a time value not in the original mapping.
        
        Parameters
        ----------
        time : float
            Time value to find the closest index for
            
        Returns
        -------
        int
            Closest index
        """
        distances = np.abs(self.time_values - time)
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def map_continuous_time(self, time: float, is_censored: bool = False) -> int:
        """Map any continuous time value to the appropriate index, with censoring awareness.
        
        For time values not in the original mapping, finds the appropriate 
        index based on where the time falls in the sorted time values.
        For censored observations, can adjust the mapping based on the censoring status.
        
        Parameters
        ----------
        time : float
            Time value to map
        is_censored : bool, optional
            Whether the observation is censored
            
        Returns
        -------
        int
            Mapped index
        """
        if time in self.time_to_idx:
            return self.time_to_idx[time]
            
        # Handle time value not in original mapping
        if time < self.time_values[0]:
            return 0
        elif time > self.time_values[-1]:
            return len(self.time_values) - 1
        else:
            # Find the insertion point
            idx = np.searchsorted(self.time_values, time)
            
            # Specialized handling for censored observations
            if self.handle_censoring and is_censored:
                # For right-censored data, we know the event hasn't happened by the censoring time
                # We might want to place it at the exact or just before censoring time
                # depending on the specific modeling approach
                return idx - 1  # Just before the insertion point
            else:
                # Standard handling for non-censored observations
                # Determine if we should use the index before or after based on proximity
                if idx > 0:
                    if (time - self.time_values[idx-1]) <= (self.time_values[idx] - time):
                        return idx - 1
                    else:
                        return idx
                return idx
                
    def handle_censored_time(self, 
                           time: float, 
                           is_censored: bool, 
                           return_idx: bool = True) -> Union[float, int]:
        """Special handler for censored time values.
        
        When dealing with censored data, we often need to treat the time values
        differently based on whether they are censored or not. This method
        provides a specialized handling for such cases.
        
        Parameters
        ----------
        time : float
            The observed time value
        is_censored : bool
            Whether the observation is censored
        return_idx : bool, optional
            Whether to return the index (True) or the time value (False)
            
        Returns
        -------
        Union[float, int]
            Mapped time index or adjusted time value
        """
        if not self.handle_censoring or not is_censored:
            # Standard handling for non-censored or when censoring handling is disabled
            if return_idx:
                return self.to_idx(time)
            else:
                return time
        
        # Special handling for censored observations
        if return_idx:
            # For indices, we generally want the exact observed time
            # But might want to adjust for specific applications
            return self.map_continuous_time(time, is_censored=True)
        else:
            # For time values, we might want to use the observed time
            # or adjust it based on domain knowledge
            return time