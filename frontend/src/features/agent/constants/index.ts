import { InternetSearchConfig } from '../types';

export const DEFAULT_INTERNET_SEARCH_CONFIG: InternetSearchConfig = {
  maxResults: 10,
};

export const EDGE_INTERNET_SEARCH_CONFIG = {
  maxResults: {
    MIN: 1,
    MAX: 50,
    STEP: 1,
  },
};
