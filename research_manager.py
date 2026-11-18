from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
import asyncio
import time
from contextlib import contextmanager

# Local trace implementation replacement for free Gemini
@contextmanager
def trace(span_name: str, trace_id: str = None):
    print(f"\n🔍 [TRACE START]: '{span_name}' (ID: {trace_id or 'local'})")
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"🔍 [TRACE END]: '{span_name}' completed in {elapsed:.4f}s\n")

def gen_trace_id() -> str:
    return "local-gemini-trace-id"

class ResearchManager:

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            yield f"Starting research. Local Trace ID: {trace_id}"
            search_plan = await self.plan_searches(query)
            yield f"Searches planned, starting {len(search_plan.searches)} searches..."     
            search_results = await self.perform_searches(search_plan)
            yield "Searches complete, writing report..."
            report = await self.write_report(query, search_results)
            yield "Report written, sending email..."
            await self.send_email(report)
            yield "Email sent, research complete"
            yield report.markdown_report

    async def plan_searches(self, query: str) -> WebSearchPlan:
        """ Plan the searches to perform for the query """
        # Assuming Runner is imported or available from writer_agent/search_agent context
        from writer_agent import Runner
        result = await Runner.run(planner_agent, f"Query: {query}")
        return result.final_output

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """ Perform the searches to perform for the query """
        tasks = [self.search(item) for item in search_plan.searches]
        return await asyncio.gather(*tasks)

    async def search(self, item: WebSearchItem) -> str | None:
        """ Perform a search for the query """
        from writer_agent import Runner
        input_message = f"Search term: {item.query}\nReason for searching: {item.reason}"
        result = await Runner.run(search_agent, input_message)
        return result.final_output

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Write the report for the query """
        from writer_agent import Runner
        input_message = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(writer_agent, input_message)
        return result.final_output
    
    async def send_email(self, report: ReportData) -> None:
        from writer_agent import Runner
        await Runner.run(email_agent, report.markdown_report)