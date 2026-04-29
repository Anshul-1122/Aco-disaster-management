import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SimulationPage from "@/pages/SimulationPage";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SimulationPage />
    </QueryClientProvider>
  );
}

export default App;
